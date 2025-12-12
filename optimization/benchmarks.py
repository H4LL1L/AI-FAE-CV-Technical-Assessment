import argparse
import json
import os
import time
from typing import Dict, Tuple, List, Optional

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import pynvml


def init_gpu_monitor(gpu_index: int = 0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return handle


def get_gpu_util(handle) -> int:
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu


def get_gpu_mem(handle) -> Dict[str, float]:
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mb = mem.used / (1024 * 1024)
    total_mb = mem.total / (1024 * 1024)
    return {
        "used_mb": float(used_mb),
        "total_mb": float(total_mb),
        "used_percent": float((used_mb / total_mb * 100.0) if total_mb > 0 else 0.0),
    }


def allocate_buffers(engine, context, input_shape: Tuple[int, int, int, int]):
    """
    TensorRT 10.x API:
      - engine.num_io_tensors
      - engine.get_tensor_name(i)
      - engine.get_tensor_mode(name)
      - engine.get_tensor_dtype(name)
      - context.set_input_shape(name, shape)
      - context.get_tensor_shape(name)
      - context.set_tensor_address(name, ptr)
      - context.execute_async_v3(stream_handle)
    """
    stream = cuda.Stream()

    num_tensors = engine.num_io_tensors

    # Input tensor isimlerini bul
    input_names: List[str] = []
    for i in range(num_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)

    if not input_names:
        raise RuntimeError("No INPUT tensors found in engine")

    # Dinamik shape'leri set et
    for name in input_names:
        ok = context.set_input_shape(name, input_shape)
        if not ok:
            raise RuntimeError(
                f"Failed to set_input_shape for tensor: {name} (shape={input_shape}); "
                "engine optimization profile may not cover this shape"
            )

    d_input = None
    input_dtype = None
    output_device_buffers: List[cuda.DeviceAllocation] = []
    host_outputs: List[np.ndarray] = []
    output_shapes: List[Tuple[int, ...]] = []

    # Tüm IO tensörleri için device buffer ayır ve adres bağla
    for i in range(num_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)

        shape = tuple(context.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        if any(dim < 0 for dim in shape):
            raise RuntimeError(
                f"Dynamic dimension still -1 after set_input_shape for tensor {name}: {shape}"
            )

        vol = int(np.prod(shape, dtype=np.int64))
        nbytes = vol * np.dtype(dtype).itemsize

        d_buf = cuda.mem_alloc(nbytes)

        # TRT 10: tensor adresini context'e set etmemiz gerekiyor
        context.set_tensor_address(name, int(d_buf))

        if mode == trt.TensorIOMode.INPUT:
            # İlk input'u "asıl" input sayıyoruz
            if d_input is None:
                d_input = d_buf
                input_dtype = dtype
        else:
            output_device_buffers.append(d_buf)
            host_arr = np.empty(shape, dtype=dtype)
            host_outputs.append(host_arr)
            output_shapes.append(shape)

    if d_input is None or input_dtype is None:
        raise RuntimeError("Failed to identify main input tensor")

    return (
        d_input,              # DeviceAllocation
        output_device_buffers,
        stream,
        host_outputs,
        output_shapes,
        input_dtype,
    )


def synthetic_preprocess(input_shape: Tuple[int, int, int, int], dtype) -> np.ndarray:
    b, c, h, w = input_shape
    # Generate uint8-like pixels and normalize to 0..1.
    pix = np.random.randint(0, 256, size=(b, c, h, w), dtype=np.uint8)
    arr = pix.astype(np.float32) / 255.0
    return arr.astype(dtype, copy=False)


def synthetic_postprocess(output: np.ndarray, shape: Tuple[int, ...]):
    # Çıkışı gerçek anlamda kullanmıyoruz, sadece CPU postprocess latency ölçüyoruz
    out = output.reshape(shape).astype(np.float32, copy=False)
    out = out.reshape(out.shape[0], -1)
    _ = out.argmax(axis=1)
    return _


def benchmark_engine(
    engine_path: str,
    precision: str,
    input_shape: Tuple[int, int, int, int],
    num_warmup: int = 10,
    num_iters: int = 100,
    gpu_index: int = 0,
) -> Dict:
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")

    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    (
        d_input,
        d_outputs,
        stream,
        host_outputs,
        output_shapes,
        input_dtype,
    ) = allocate_buffers(engine, context, input_shape)

    batch_size = input_shape[0]
    gpu_handle = init_gpu_monitor(gpu_index)

    gpu_times = []
    gpu_exec_times = []
    cpu_pre_times = []
    cpu_post_times = []
    gpu_utils = []
    gpu_mem_used_mb = []
    gpu_mem_used_pct = []

    start_evt = cuda.Event()
    end_evt = cuda.Event()

    # Warmup
    for _ in range(num_warmup):
        host_input = synthetic_preprocess(input_shape, input_dtype)
        cuda.memcpy_htod_async(d_input, host_input, stream)

        ok = context.execute_async_v3(stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("Failed to execute_async_v3 in warmup")

        for i, d_out in enumerate(d_outputs):
            cuda.memcpy_dtoh_async(host_outputs[i], d_out, stream)
        stream.synchronize()

    # Timed runs
    for _ in range(num_iters):
        t0 = time.perf_counter()
        host_input = synthetic_preprocess(input_shape, input_dtype)
        t1 = time.perf_counter()

        cuda.memcpy_htod_async(d_input, host_input, stream)
        start_evt.record(stream)
        ok = context.execute_async_v3(stream_handle=stream.handle)
        end_evt.record(stream)
        if not ok:
            raise RuntimeError("Failed to execute_async_v3 in timed run")

        for i, d_out in enumerate(d_outputs):
            cuda.memcpy_dtoh_async(host_outputs[i], d_out, stream)
        stream.synchronize()
        t2 = time.perf_counter()

        # İlk output tensor üzerinde "postprocess"
        _ = synthetic_postprocess(host_outputs[0], output_shapes[0])
        t3 = time.perf_counter()

        cpu_pre_times.append((t1 - t0) * 1000.0)
        # Includes H2D + execute + D2H + stream sync.
        gpu_times.append((t2 - t1) * 1000.0)
        # Pure GPU execute time only (no D2H) because end event is recorded before copies.
        gpu_exec_times.append(start_evt.time_till(end_evt))
        cpu_post_times.append((t3 - t2) * 1000.0)

        gpu_utils.append(get_gpu_util(gpu_handle))
        mem = get_gpu_mem(gpu_handle)
        gpu_mem_used_mb.append(mem["used_mb"])
        gpu_mem_used_pct.append(mem["used_percent"])

    def stats(arr):
        arr_np = np.array(arr, dtype=np.float32)
        return {
            "avg_ms": float(arr_np.mean()),
            "p50_ms": float(np.percentile(arr_np, 50)),
            "p95_ms": float(np.percentile(arr_np, 95)),
        }

    gpu_stats = stats(gpu_times)
    gpu_exec_stats = stats(gpu_exec_times)
    pre_stats = stats(cpu_pre_times)
    post_stats = stats(cpu_post_times)

    total_infer_time_s = sum(gpu_times) / 1000.0
    throughput = (num_iters * batch_size) / total_infer_time_s if total_infer_time_s > 0 else 0.0
    avg_gpu_util = float(np.mean(gpu_utils)) if gpu_utils else 0.0

    result = {
        "engine_path": engine_path,
        "precision": precision,
        "input_shape": input_shape,
        "batch_size": batch_size,
        "num_iterations": num_iters,
        "warmup_iterations": num_warmup,
        "latency_ms": {
            # "gpu_total" includes H2D/D2H transfers and sync on the same stream.
            "gpu_total": gpu_stats,
            # "gpu_execute" tries to isolate TRT execution time using CUDA events.
            "gpu_execute": gpu_exec_stats,
            "cpu_pre": pre_stats,
            "cpu_post": post_stats,
        },
        "throughput_fps": throughput,
        "gpu_utilization": {
            "avg_percent": avg_gpu_util,
            "mem_used_mb_avg": float(np.mean(gpu_mem_used_mb)) if gpu_mem_used_mb else 0.0,
            "mem_used_mb_max": float(np.max(gpu_mem_used_mb)) if gpu_mem_used_mb else 0.0,
            "mem_used_percent_avg": float(np.mean(gpu_mem_used_pct)) if gpu_mem_used_pct else 0.0,
        },
        "notes": {
            "gpu_total_includes_h2d_d2h": True,
        },
    }
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engines (TRT 10.x)")
    parser.add_argument(
        "--fp16_engine",
        default="models/last_fp16.engine",
        help="Path to FP16 engine",
    )
    parser.add_argument(
        "--int8_engine",
        default="models/last_int8.engine",
        help="Path to INT8 engine",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to benchmark",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Input height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Input width",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="GPU index for NVML metrics",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_shape = (args.batch_size, 3, args.height, args.width)

    print("[INFO] Benchmarking FP16 engine...")
    fp16_results = benchmark_engine(
        engine_path=args.fp16_engine,
        precision="fp16",
        input_shape=input_shape,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        gpu_index=args.gpu_index,
    )

    print("[INFO] Benchmarking INT8 engine...")
    int8_results = benchmark_engine(
        engine_path=args.int8_engine,
        precision="int8",
        input_shape=input_shape,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        gpu_index=args.gpu_index,
    )

    all_results = {"fp16": fp16_results, "int8": int8_results}

    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[INFO] Benchmark results saved to {out_path}")
