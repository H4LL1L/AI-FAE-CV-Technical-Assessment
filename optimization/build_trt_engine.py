import argparse
import os
from typing import Tuple

import tensorrt as trt


class CacheCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Sadece mevcut calibration cache'i okuyan calibrator.
    Final INT8 engine build ederken kullanıyoruz.
    """

    def __init__(self, cache_file: str, batch_size: int, input_hw: Tuple[int, int]):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_hw = input_hw

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # Yeni veriyle kalibrasyon yapmıyoruz, sadece cache kullanıyoruz
        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[INFO] Loading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        print("[WARN] Calibration cache file not found, INT8 build may fail")
        return None

    def write_calibration_cache(self, cache):
        # Cache yazma işi calibrate_int8.py'de yapıldı, burada yok.
        pass


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str,
    calibration_cache: str = "",
    max_batch_size: int = 8,
    workspace_size_mb: int = 2048,
):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    print(f"[INFO] Building engine:")
    print(f"       ONNX:   {onnx_path}")
    print(f"       OUT:    {engine_path}")
    print(f"       PREC:   {precision}")
    print(f"       BATCH:  <= {max_batch_size}")
    print(f"       WORKSPACE: {workspace_size_mb} MB")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_size_mb * (1 << 20)
    )

    input_tensor = network.get_input(0)

    min_shape = (1, 3, 384, 384)
    opt_batch = min(4, max_batch_size)
    opt_shape = (opt_batch, 3, 640, 640)
    max_shape = (max_batch_size, 3, 960, 960)

    print(f"[INFO] Optimization profile:")
    print(f"       MIN: {min_shape}")
    print(f"       OPT: {opt_shape}")
    print(f"       MAX: {max_shape}")

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if precision.lower() == "fp16":
        print("[INFO] Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision.lower() == "int8":
        print("[INFO] Enabling INT8 mode")
        config.set_flag(trt.BuilderFlag.INT8)
        if not calibration_cache:
            raise ValueError("INT8 requested but no calibration cache provided")
        calibrator = CacheCalibrator(
            cache_file=calibration_cache,
            batch_size=opt_batch,
            input_hw=(opt_shape[2], opt_shape[3]),
        )
        config.int8_calibrator = calibrator
    else:
        print("[INFO] Building FP32 engine (no precision flag)")

    print(f"[INFO] Building {precision.upper()} TensorRT engine...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"Failed to build {precision.upper()} TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    print(f"[INFO] Saved engine to {engine_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX")
    parser.add_argument(
        "--onnx_path",
        default="models/last.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--fp16_engine",
        default="models/last_fp16.engine",
        help="Output path for FP16 engine",
    )
    parser.add_argument(
        "--int8_engine",
        default="models/last_int8.engine",
        help="Output path for INT8 engine",
    )
    parser.add_argument(
        "--calibration_cache",
        default="models/calibration.cache",
        help="Path to calibration cache file",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size",
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

    print("[INFO] ===== FP16 ENGINE BUILD =====")
    build_engine(
        onnx_path=args.onnx_path,
        engine_path=args.fp16_engine,
        precision="fp16",
        max_batch_size=args.max_batch_size,
        workspace_size_mb=args.workspace_size_mb,
    )

    print("[INFO] ===== INT8 ENGINE BUILD =====")
    build_engine(
        onnx_path=args.onnx_path,
        engine_path=args.int8_engine,
        precision="int8",
        calibration_cache=args.calibration_cache,
        max_batch_size=args.max_batch_size,
        workspace_size_mb=args.workspace_size_mb,
    )

    print("[INFO] All engines built.")
