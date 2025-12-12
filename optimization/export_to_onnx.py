import os
import logging

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO


LOG_PATH = "models/log_export_onnx.txt"


def setup_logger():
    logger = logging.getLogger("export_onnx")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    os.makedirs("models", exist_ok=True)

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


logger = setup_logger()


def export_pytorch_to_onnx(
    weights_path: str = "models/last.pt",
    onnx_path: str = "models/last.onnx",
    img_size: int = 640,
    opset: int = 12,
):
    """
    Ultralytics YOLO PyTorch modeli ONNX'e export eder:
    - Dinamik batch
    - Dinamik height/width
    ONNX çıktısını PyTorch ile karşılaştırır.
    Tüm loglar: models/log_export_onnx.txt
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if not os.path.exists(weights_path):
        logger.error(f"PyTorch weights not found: {weights_path}")
        raise FileNotFoundError(f"PyTorch weights not found: {weights_path}")

    logger.info(f"Loading YOLO model from {weights_path}")
    yolo = YOLO(weights_path)
    model = yolo.model
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    logger.info(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["images"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        },
    )

    logger.info("Export done, running ONNX validation...")

    # Validation
    dummy_np = dummy_input.detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        pt_out = model(dummy_input)
        if isinstance(pt_out, (list, tuple)):
            pt_out = pt_out[0]
        pt_np = pt_out.detach().cpu().numpy()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: dummy_np})[0]

    pt_flat = pt_np.astype(np.float32).reshape(-1)
    onnx_flat = onnx_out.astype(np.float32).reshape(-1)

    min_len = min(pt_flat.shape[0], onnx_flat.shape[0])
    diff = np.abs(pt_flat[:min_len] - onnx_flat[:min_len])

    logger.info(f"Validation mean diff={diff.mean():.6f}")
    logger.info(f"Validation max  diff={diff.max():.6f}")

    logger.info("PyTorch → ONNX export & validation completed.")
    logger.info(f"Log saved to {LOG_PATH}")


if __name__ == "__main__":
    export_pytorch_to_onnx()
