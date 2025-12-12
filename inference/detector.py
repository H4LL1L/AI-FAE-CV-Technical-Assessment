"""
Multi-backend detector wrapper with:
- Ultralytics path (PyTorch / ORT / TensorRT via YOLO)
- Native ONNX Runtime (CPU/CUDA/TensorRT EP) with custom pre/post

Usage:
    det = Detector("models/model.onnx", backend="ort", device="cpu", imgsz=640)
    result = det(frame_bgr)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .utils import Detection, classwise_nms, now_ms, iou_matrix

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    trt = None
    cuda = None


def _load_dataset_names() -> Optional[Dict[int, str]]:
    """
    Try to load class names from training/dataset.yaml to avoid defaulting to 'person'.
    """
    ds_path = Path(__file__).resolve().parents[1] / "training" / "dataset.yaml"
    if not ds_path.exists():
        return None
    try:
        data = yaml.safe_load(ds_path.read_text())
        names = data.get("names")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        return None
    return None


def _fallback_names(model: Any) -> None:
    """
    TensorRT engines sometimes lose metadata. Patch task/names defensively.
    """
    fallback = _load_dataset_names() or {0: "object"}
    backend = getattr(model, "model", None)
    if backend is None:
        return

    try:
        if not getattr(backend, "task", None):
            backend.__dict__["task"] = "detect"  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        names = getattr(backend, "names", None)
    except Exception:
        names = None
    if not names or len(names) == 0 or len(names) == 999:
        try:
            backend.names = fallback  # type: ignore[assignment]
        except Exception:
            pass


def _letterbox(
    img: np.ndarray, new_shape: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize + pad to keep aspect ratio, similar to Ultralytics letterbox.
    Returns image, scale factor, and padding (pad_w, pad_h).
    """
    shape = img.shape[:2]  # (h, w)
    if not isinstance(new_shape, tuple):
        new_shape = (int(new_shape), int(new_shape))

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.zeros_like(xywh, dtype=np.float32)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out


def _postprocess_yolo_preds(
    preds: np.ndarray,
    conf: float,
    iou: float,
    classes: Optional[List[int]],
    names: Dict[int, str],
    use_custom_nms: bool,
    ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int],
    max_det: int,
) -> List[Detection]:
    """
    Shared postprocess for YOLO-style outputs.

    Supports both common head formats:
      - [x, y, w, h, cls_scores...]
      - [x, y, w, h, obj, cls_scores...]  (confidence = obj * cls_score)

    Handles shapes (1,N,C), (N,C), or (C,N) and returns Detection list.
    """
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:  # (C, N)
        preds = preds.transpose(1, 0)

    if preds.shape[1] < 5:
        return []

    boxes = preds[:, :4]

    # Infer whether an "objectness" column exists using class-count when possible.
    # When training/dataset.yaml is present, this is usually reliable.
    expected_classes: int | None = None
    try:
        keys = [int(k) for k in names.keys()]
        if keys:
            expected_classes = max(keys) + 1
    except Exception:
        expected_classes = None

    has_objectness: bool | None = None
    if expected_classes is not None and expected_classes > 0:
        if preds.shape[1] == 5 + expected_classes:
            has_objectness = True
        elif preds.shape[1] == 4 + expected_classes:
            has_objectness = False

    if has_objectness is None:
        # Heuristic fallback when class-count cannot be inferred.
        common_class_counts = {1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 16, 20, 40, 60, 80, 90, 100}
        if preds.shape[1] >= 6 and (preds.shape[1] - 5) in common_class_counts and (preds.shape[1] - 4) not in common_class_counts:
            has_objectness = True
        else:
            has_objectness = False

    if has_objectness:
        if preds.shape[1] < 6:
            return []
        obj = preds[:, 4:5]
        scores_cls = preds[:, 5:]
        if scores_cls.size == 0:
            return []
        scores_cls = scores_cls * obj
    else:
        scores_cls = preds[:, 4:]
        if scores_cls.size == 0:
            return []

    cls_idx = np.argmax(scores_cls, axis=1)
    cls_scores = scores_cls[np.arange(scores_cls.shape[0]), cls_idx]

    mask = cls_scores >= conf
    if classes is not None:
        mask = mask & np.isin(cls_idx, classes)
    if not np.any(mask):
        return []

    boxes = boxes[mask]
    cls_idx = cls_idx[mask]
    cls_scores = cls_scores[mask]

    xyxy = _xywh_to_xyxy(boxes)
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy[:, :4] /= ratio

    h, w = orig_shape
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h)

    if use_custom_nms:
        keep = classwise_nms(xyxy, cls_scores, cls_idx, iou_thr=iou)
        xyxy = xyxy[keep]
        cls_scores = cls_scores[keep]
        cls_idx = cls_idx[keep]

    if max_det > 0 and len(cls_scores) > max_det:
        order = np.argsort(-cls_scores)[:max_det]
        xyxy = xyxy[order]
        cls_scores = cls_scores[order]
        cls_idx = cls_idx[order]

    dets: List[Detection] = []
    for i in range(len(xyxy)):
        label = names.get(int(cls_idx[i]), None)
        dets.append(
            Detection(
                xyxy=xyxy[i],
                score=float(cls_scores[i]),
                cls=int(cls_idx[i]),
                label=label,
            )
        )
    return dets


class _UltralyticsRunner:
    def __init__(
        self,
        model_path: Path,
        device: str,
        imgsz: int,
        conf: float,
        iou: float,
        classes: Optional[List[int]],
        max_det: int,
        use_custom_nms: bool,
    ) -> None:
        from ultralytics import YOLO  # local import to avoid torch dependency when unused

        self.model = YOLO(str(model_path))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.max_det = max_det
        self.use_custom_nms = use_custom_nms
        self.backend_name = self._infer_backend_name(model_path)

        if model_path.suffix.lower() == ".engine":
            _fallback_names(self.model)

        self.names = getattr(self.model.model, "names", None) or {0: "person"}

    def _infer_backend_name(self, model_path: Path) -> str:
        suf = model_path.suffix.lower()
        if suf == ".pt":
            return "pytorch"
        if suf == ".onnx":
            return "onnxruntime-ultralytics"
        if suf == ".engine":
            return "tensorrt-ultralytics"
        return "ultralytics"

    def warmup(self, iters: int, imgsz: int) -> None:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(iters):
            try:
                self.model.predict(
                    dummy,
                    imgsz=imgsz,
                    device=self.device,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    classes=self.classes,
                    verbose=False,
                )
            except Exception:
                break

    def run(self, batch_list: List[np.ndarray]) -> Tuple[List[List[Detection]], float]:
        t0 = now_ms()
        results = self.model.predict(
            batch_list,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
        )
        t1 = now_ms()

        detections_batch: List[List[Detection]] = []
        for res in results or []:
            boxes = getattr(res, "boxes", None)
            dets: List[Detection] = []
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                if self.use_custom_nms:
                    keep_idx = classwise_nms(xyxy, scores, cls, iou_thr=self.iou)
                    xyxy = xyxy[keep_idx]
                    scores = scores[keep_idx]
                    cls = cls[keep_idx]

                if self.max_det > 0 and len(scores) > self.max_det:
                    order = np.argsort(-scores)[: self.max_det]
                    xyxy = xyxy[order]
                    scores = scores[order]
                    cls = cls[order]

                for i in range(len(xyxy)):
                    label = None
                    try:
                        label = self.names.get(int(cls[i]), None)  # type: ignore[arg-type]
                    except Exception:
                        label = None
                    dets.append(
                        Detection(
                            xyxy=xyxy[i],
                            score=float(scores[i]),
                            cls=int(cls[i]),
                            label=label,
                        )
                    )
            detections_batch.append(dets)
        return detections_batch, t1 - t0


class _ORTRunner:
    """
    Native ONNX Runtime runner with simple letterbox pre/post and NMS.
    """

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        providers: List[str],
        use_custom_nms: bool,
        classes: Optional[List[int]] = None,
    ) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for backend='ort'")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.use_custom_nms = use_custom_nms
        self.classes = classes
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # Try to fetch names metadata if present, else fallback to dataset.yaml
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        if meta:
            names_meta: Dict[int, str] = {}
            for k, v in meta.items():
                try:
                    names_meta[int(k)] = str(v)
                except Exception:
                    continue
            self.names = names_meta or (_load_dataset_names() or {0: "object"})
        else:
            self.names = _load_dataset_names() or {0: "object"}

    def warmup(self, iters: int, imgsz: int) -> None:
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        for _ in range(iters):
            try:
                _ = self.session.run(None, {self.input_name: dummy})
            except Exception:
                break

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lb, r, (dw, dh) = _letterbox(img, (self.imgsz, self.imgsz))
        lb = lb.astype(np.float32) / 255.0
        chw = np.transpose(lb, (2, 0, 1))
        return chw[None, ...], r, (dw, dh), img.shape[:2]

    def _postprocess(
        self,
        preds: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        return _postprocess_yolo_preds(
            preds=preds,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            names=self.names,
            use_custom_nms=self.use_custom_nms,
            ratio=ratio,
            pad=pad,
            orig_shape=orig_shape,
            max_det=self.max_det,
        )

    def run(self, batch_list: List[np.ndarray]) -> Tuple[List[List[Detection]], float]:
        t0 = now_ms()
        if not batch_list:
            return [], 0.0

        inputs: List[np.ndarray] = []
        metas: List[Tuple[float, Tuple[float, float], Tuple[int, int]]] = []
        for frame in batch_list:
            inp, r, pad, orig_shape = self._preprocess(frame)
            inputs.append(inp)
            metas.append((r, pad, orig_shape))

        outputs: List[List[Detection]] = []
        if len(inputs) == 1:
            ort_out = self.session.run(None, {self.input_name: inputs[0]})[0]
            outputs.append(self._postprocess(ort_out, metas[0][0], metas[0][1], metas[0][2]))
        else:
            batch_inp = np.concatenate(inputs, axis=0)
            try:
                ort_out = self.session.run(None, {self.input_name: batch_inp})[0]
                if ort_out.ndim == 2:
                    preds_batch = [ort_out]
                elif ort_out.ndim == 3 and ort_out.shape[0] == len(inputs):
                    preds_batch = [ort_out[i] for i in range(ort_out.shape[0])]
                elif ort_out.ndim == 3 and ort_out.shape[-1] == len(inputs):
                    # Unusual layout (N,C,B) -> (B,N,C)
                    ort_out2 = np.moveaxis(ort_out, -1, 0)
                    preds_batch = [ort_out2[i] for i in range(ort_out2.shape[0])]
                else:
                    raise RuntimeError(f"Unexpected ORT output shape for batch: {ort_out.shape}")
                for pred, (r, pad, orig_shape) in zip(preds_batch, metas):
                    outputs.append(self._postprocess(pred, r, pad, orig_shape))
            except Exception:
                for inp, (r, pad, orig_shape) in zip(inputs, metas):
                    ort_out = self.session.run(None, {self.input_name: inp})[0]
                    outputs.append(self._postprocess(ort_out, r, pad, orig_shape))
        t1 = now_ms()
        return outputs, t1 - t0


class _TRTRunner:
    """
    Native TensorRT runner using Python API + PyCUDA.
    Provides the same preprocess/postprocess as ORT runner for consistency.
    """

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        use_custom_nms: bool,
        classes: Optional[List[int]] = None,
    ) -> None:
        if trt is None or cuda is None:
            raise ImportError("TensorRT and pycuda are required for backend='tensorrt'")

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.use_custom_nms = use_custom_nms
        self.classes = classes
        self.names = _load_dataset_names() or {0: "object"}

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        runtime = trt.Runtime(self.logger)
        with open(model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._use_v3 = hasattr(self.context, "execute_async_v3") and hasattr(self.context, "set_tensor_address")

        self._input_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
                break
        if self._input_name is None:
            raise RuntimeError("No INPUT tensor found in TensorRT engine")

        self._bindings: List[int] = []
        self._device_input: Optional[cuda.DeviceAllocation] = None
        self._device_outputs: List[cuda.DeviceAllocation] = []
        self._host_outputs: List[np.ndarray] = []
        self._input_dtype = trt.nptype(self.engine.get_tensor_dtype(self._input_name))
        self._cached_bs: Optional[int] = None

    def _allocate(self, batch_size: int) -> None:
        if self._cached_bs == batch_size and self._device_input is not None:
            return

        shape = (batch_size, 3, self.imgsz, self.imgsz)
        ok = self.context.set_input_shape(self._input_name, shape)
        if not ok:
            raise RuntimeError(f"Failed to set input shape {shape} for {self._input_name}")

        self._bindings = []
        self._device_outputs = []
        self._host_outputs = []
        self._device_input = None

        if not self._use_v3:
            if hasattr(self.engine, "num_bindings") and hasattr(self.engine, "get_binding_index"):
                self._bindings = [0] * int(self.engine.num_bindings)
            else:
                self._bindings = [0] * self.engine.num_io_tensors

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            tensor_shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if any(dim < 0 for dim in tensor_shape):
                raise RuntimeError(f"Dynamic dimension unresolved for tensor {name}: {tensor_shape}")

            vol = int(np.prod(tensor_shape, dtype=np.int64))
            nbytes = vol * np.dtype(dtype).itemsize
            d_buf = cuda.mem_alloc(nbytes)

            if self._use_v3:
                # TensorRT "tensor API" path: bind by name, not by legacy binding order.
                self.context.set_tensor_address(name, int(d_buf))
            else:
                # Legacy bindings list: keep correct binding index mapping when available.
                if hasattr(self.engine, "get_binding_index") and hasattr(self.engine, "num_bindings"):
                    bidx = int(self.engine.get_binding_index(name))
                    self._bindings[bidx] = int(d_buf)
                else:
                    self._bindings[i] = int(d_buf)

            if mode == trt.TensorIOMode.INPUT:
                self._device_input = d_buf
            else:
                self._device_outputs.append(d_buf)
                self._host_outputs.append(np.empty(tensor_shape, dtype=dtype))

        if self._device_input is None:
            raise RuntimeError("Input buffer allocation failed for TensorRT runner")

        self._cached_bs = batch_size

    def _preprocess_batch(
        self, batch_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[float, Tuple[float, float], Tuple[int, int]]]]:
        inputs: List[np.ndarray] = []
        metas: List[Tuple[float, Tuple[float, float], Tuple[int, int]]] = []
        for frame in batch_list:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lb, r, pad = _letterbox(img, (self.imgsz, self.imgsz))
            lb = lb.astype(np.float32) / 255.0
            chw = np.transpose(lb, (2, 0, 1))
            inputs.append(chw)
            metas.append((r, pad, frame.shape[:2]))
        arr = np.stack(inputs, axis=0).astype(self._input_dtype, copy=False)
        return arr, metas

    def warmup(self, iters: int, imgsz: int) -> None:
        try:
            dummy = np.zeros((1, 3, imgsz, imgsz), dtype=self._input_dtype)
            self._allocate(batch_size=1)
            for _ in range(iters):
                cuda.memcpy_htod_async(self._device_input, dummy, self.stream)
                if self._use_v3:
                    self.context.execute_async_v3(self.stream.handle)
                else:
                    self.context.execute_async_v2(bindings=self._bindings, stream_handle=self.stream.handle)
                for i, d_out in enumerate(self._device_outputs):
                    cuda.memcpy_dtoh_async(self._host_outputs[i], d_out, self.stream)
                self.stream.synchronize()
        except Exception:
            # Warmup is best-effort; ignore failures to keep runtime robust
            pass

    def run(self, batch_list: List[np.ndarray]) -> Tuple[List[List[Detection]], float]:
        batch_input, metas = self._preprocess_batch(batch_list)
        self._allocate(batch_size=batch_input.shape[0])

        t0 = now_ms()
        cuda.memcpy_htod_async(self._device_input, batch_input, self.stream)
        if self._use_v3:
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self._bindings, stream_handle=self.stream.handle)
        for i, d_out in enumerate(self._device_outputs):
            cuda.memcpy_dtoh_async(self._host_outputs[i], d_out, self.stream)
        self.stream.synchronize()
        t1 = now_ms()

        preds = self._host_outputs[0]
        if preds.ndim == 2:  # (N, C) for bs=1
            preds = preds[None, ...]
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            preds = preds.transpose(0, 2, 1)  # (B, N, C)

        outputs: List[List[Detection]] = []
        for pred, (ratio, pad, orig_shape) in zip(preds, metas):
            dets = _postprocess_yolo_preds(
                preds=pred,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                names=self.names,
                use_custom_nms=self.use_custom_nms,
                ratio=ratio,
                pad=pad,
                orig_shape=orig_shape,
                max_det=self.max_det,
            )
            outputs.append(dets)

        return outputs, t1 - t0


class Detector:
    """
    Unified detector with explicit backend selection.
    backend options:
      - "auto" (default): .onnx -> ORT, .engine -> TensorRT, else Ultralytics
      - "ort": native onnxruntime
      - "ort-trt": onnxruntime with TensorRT EP (if available)
      - "tensorrt": native TensorRT (Python API)
      - "ultralytics": YOLO wrapper (PT/ONNX/TRT)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        warmup_iters: int = 3,
        use_custom_nms: bool = True,
        backend: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.max_det = max_det
        self.use_custom_nms = use_custom_nms

        self.backend_mode = backend.lower()
        if self.backend_mode == "auto":
            suf = self.model_path.suffix.lower()
            if suf == ".onnx":
                self.backend_mode = "ort"
            elif suf == ".engine":
                self.backend_mode = "tensorrt"
            else:
                self.backend_mode = "ultralytics"

        if self.backend_mode in ("ort", "ort-trt"):
            providers = ["CPUExecutionProvider"]
            if self.device != "cpu":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.backend_mode == "ort-trt":
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            self._runner = _ORTRunner(
                model_path=self.model_path,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                providers=providers,
                use_custom_nms=self.use_custom_nms,
                classes=self.classes,
            )
            self.backend = self.backend_mode
            self.names = getattr(self._runner, "names", {0: "person"})
        elif self.backend_mode == "tensorrt":
            self._runner = _TRTRunner(
                model_path=self.model_path,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                use_custom_nms=self.use_custom_nms,
                classes=self.classes,
            )
            self.backend = "tensorrt"
            self.names = getattr(self._runner, "names", {0: "object"})
        else:
            self._runner = _UltralyticsRunner(
                model_path=self.model_path,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                max_det=self.max_det,
                use_custom_nms=self.use_custom_nms,
            )
            self.backend = self._runner.backend_name
            self.names = self._runner.names

        # Optional warmup to stabilize first-run latency
        if warmup_iters > 0:
            try:
                self._runner.warmup(warmup_iters, self.imgsz)
            except Exception:
                pass

    def __call__(self, frames: np.ndarray | Sequence[np.ndarray]) -> Dict[str, object]:
        """
        Run detection on one frame or a batch (list/4D array of BGR images).

        Returns:
            {
                "detections": List[Detection] or List[List[Detection]] (if batch),
                "timings_ms": {"inference": float, "total": float},
                "backend": str,
            }
        """
        if frames is None:
            raise ValueError("Input frame is empty")

        # Normalize input to list for batching
        is_batch = isinstance(frames, (list, tuple)) or (
            isinstance(frames, np.ndarray) and frames.ndim == 4
        )
        batch_list: List[np.ndarray]
        if is_batch:
            if isinstance(frames, np.ndarray):
                batch_list = [f for f in frames]
            else:
                batch_list = list(frames)
        else:
            frame_bgr = frames  # type: ignore[assignment]
            if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                raise ValueError("Input frame is empty")
            batch_list = [frame_bgr]

        dets_batch, infer_ms = self._runner.run(batch_list)
        out_dets: List[Detection] | List[List[Detection]]
        if is_batch:
            out_dets = dets_batch
        else:
            out_dets = dets_batch[0] if dets_batch else []
        return {
            "detections": out_dets,
            "timings_ms": {"inference": infer_ms, "total": infer_ms},
            "backend": self.backend,
        }


def load_opencv_image(path: str) -> np.ndarray:
    """Convenience loader that returns a BGR image."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


# ByteTrack-lite implementation (previously in inference/bytetrack.py) for convenience
_bt_id_gen = itertools.count(1)


@dataclass
class BTTrack:
    track_id: int
    xyxy: np.ndarray
    cls: int
    score: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0

    def to_detection(self, label_lookup: dict | None = None) -> Detection:
        label = label_lookup.get(self.cls) if label_lookup else None
        return Detection(
            xyxy=self.xyxy.copy(),
            score=self.score,
            cls=self.cls,
            label=label,
            track_id=self.track_id,
        )


class ByteTrackLite:
    """
    Lightweight ByteTrack-inspired tracker.
    Two-stage association: high-score then low-score detections.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_thresh_high: float = 0.5,
        iou_thresh_low: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh_high = iou_thresh_high
        self.iou_thresh_low = iou_thresh_low
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[BTTrack] = []

    def _create(self, det: Detection) -> BTTrack:
        return BTTrack(
            track_id=next(_bt_id_gen),
            xyxy=det.xyxy.astype(np.float32),
            cls=det.cls,
            score=det.score,
        )

    def step(self, detections: List[Detection], label_lookup: dict | None = None) -> List[Detection]:
        """Update with detections; returns active tracks as detections."""
        if len(detections) == 0:
            return self._age_and_prune(label_lookup)

        det_high = [d for d in detections if d.score >= self.high_thresh]
        det_low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        # Stage 1: match high-score dets
        matched_high = set()
        matched_tracks = set()
        if self.tracks and det_high:
            ious = iou_matrix(np.stack([t.xyxy for t in self.tracks]), np.stack([d.xyxy for d in det_high]))
            while True:
                if ious.size == 0:
                    break
                r, c = np.unravel_index(np.argmax(ious), ious.shape)
                if ious[r, c] < self.iou_thresh_high:
                    break
                if r in matched_tracks or c in matched_high:
                    ious[r, c] = -1
                    continue
                trk = self.tracks[r]
                det = det_high[c]
                trk.xyxy = det.xyxy.astype(np.float32)
                trk.score = det.score
                trk.cls = det.cls
                trk.hits += 1
                trk.time_since_update = 0
                trk.age += 1
                matched_tracks.add(r)
                matched_high.add(c)
                ious[r, :] = -1
                ious[:, c] = -1

        # Stage 2: low-score dets try to match unmatched tracks
        if self.tracks and det_low:
            unmatched_tracks = [t for i, t in enumerate(self.tracks) if i not in matched_tracks]
            if unmatched_tracks:
                ious = iou_matrix(np.stack([t.xyxy for t in unmatched_tracks]), np.stack([d.xyxy for d in det_low]))
                while True:
                    if ious.size == 0:
                        break
                    r, c = np.unravel_index(np.argmax(ious), ious.shape)
                    if ious[r, c] < self.iou_thresh_low:
                        break
                    trk = unmatched_tracks[r]
                    det = det_low[c]
                    trk.xyxy = det.xyxy.astype(np.float32)
                    trk.score = det.score
                    trk.cls = det.cls
                    trk.hits += 1
                    trk.time_since_update = 0
                    trk.age += 1
                    ious[r, :] = -1
                    ious[:, c] = -1

        # New tracks for unmatched high-score dets
        for idx, det in enumerate(det_high):
            if idx not in matched_high:
                self.tracks.append(self._create(det))

        # Age/prune unmatched
        return self._age_and_prune(label_lookup)

    # Align interface with Tracker
    def update(self, detections: List[Detection], label_lookup: dict | None = None) -> List[Detection]:
        return self.step(detections, label_lookup)

    def step_without_detections(self) -> List[Detection]:
        return self._age_and_prune(None)

    def _age_and_prune(self, label_lookup: dict | None) -> List[Detection]:
        active: List[BTTrack] = []
        survivors: List[BTTrack] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                t.age += 1
            t.time_since_update += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
            if t.hits >= self.min_hits and t.time_since_update == 1:
                active.append(t)
        self.tracks = survivors
        return [t.to_detection(label_lookup) for t in active]
