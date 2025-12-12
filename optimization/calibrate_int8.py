import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


def preprocess_image(path: str, height: int, width: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: List[str],
        batch_size: int,
        input_shape: Tuple[int, int, int, int],
        cache_file: str,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.input_shape = input_shape  # (N, C, H, W)
        self.cache_file = cache_file
        self.current_index = 0

        # Toplam eleman sayısını PYTHON int'e çevir
        n_elems_per_sample = int(np.prod(self.input_shape[1:], dtype=np.int64))
        n_elems_total = int(batch_size * n_elems_per_sample)
        n_bytes = int(n_elems_total * np.float32().nbytes)

        self.device_input = cuda.mem_alloc(n_bytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[
            self.current_index : self.current_index + self.batch_size
        ]
        actual_batch_size = len(batch_paths)

        imgs = [
            preprocess_image(
                p, self.input_shape[2], self.input_shape[3]
            )
            for p in batch_paths
        ]

        # Son batch küçükse pad et ki her zaman sabit batch_size olsun
        if actual_batch_size < self.batch_size:
            imgs += [imgs[-1]] * (self.batch_size - actual_batch_size)

        batch = np.stack(imgs, axis=0).astype(np.float32)

        # C-contiguous hale getir (PyCUDA bunu istiyor)
        batch = np.ascontiguousarray(batch)

        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size

        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[INFO] Using existing calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[INFO] Saving calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_calibration_cache(
    onnx_path: str,
    calib_images_dir: str,
    cache_path: str,
    batch_size: int = 8,
    max_calib_images: int = 500,
    workspace_size_mb: int = 2048,
):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print(f"[INFO] Calibration images dir: {calib_images_dir}")

    image_paths = sorted(
        glob.glob(os.path.join(calib_images_dir, "*.jpg"))
        + glob.glob(os.path.join(calib_images_dir, "*.jpeg"))
        + glob.glob(os.path.join(calib_images_dir, "*.png"))
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {calib_images_dir} for calibration")

    image_paths = image_paths[:max_calib_images]
    print(f"[INFO] Using {len(image_paths)} images for INT8 calibration")

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model for calibration")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_size_mb * (1 << 20)
    )

    input_tensor = network.get_input(0)
    # Dynamic (N, C, H, W)
    min_shape = (1, 3, 384, 384)
    opt_shape = (batch_size, 3, 640, 640)
    max_shape = (batch_size, 3, 960, 960)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    input_shape = opt_shape
    calibrator = EntropyCalibrator(
        image_paths=image_paths,
        batch_size=batch_size,
        input_shape=input_shape,
        cache_file=cache_path,
    )

    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    print("[INFO] Building temporary INT8 engine for calibration...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build INT8 calibration engine")

    print("[INFO] Calibration completed, cache should be saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="INT8 calibration cache generator")
    parser.add_argument(
        "--onnx_path",
        default="models/last.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--calib_images_dir",
        default="/content/data_raw/100325_matsumoto_labelImg_temiz/images/val",
        help="Directory with calibration images",
    )
    parser.add_argument(
        "--cache_path",
        default="models/calibration.cache",
        help="Output path for calibration cache",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--max_calib_images",
        type=int,
        default=500,
        help="Max number of calibration images to use",
    )
    parser.add_argument(
        "--workspace_size_mb",
        type=int,
        default=2048,
        help="TensorRT workspace size in MB",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_calibration_cache(
        onnx_path=args.onnx_path,
        calib_images_dir=args.calib_images_dir,
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        max_calib_images=args.max_calib_images,
        workspace_size_mb=args.workspace_size_mb,
    )
