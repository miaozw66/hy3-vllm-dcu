#!/usr/bin/env python3
"""Measure real peak FP16/INT8 TFLOPS and HBM bandwidth on the Hygon DCU.

Runs on the (idle) machine and uses torch primitives so the numbers reflect
what the software stack can actually achieve — NOT vendor spec. This is the
denominator for the roofline utilization analysis (see ROOFLINE_8K report).

Outputs:
  - FP16 GEMM peak TFLOPS  (torch.matmul, M=N=K=8192)
  - INT8 GEMM peak TOPS    (torch._int_mm when the build supports it)
  - HBM copy bandwidth     (big-tensor y.copy_(x), GB/s)
  - HBM read bandwidth     (big-tensor .sum(), GB/s)
  - sclk / mclk sampled under load (rocm-smi) — verifies whether the idle
    mclk=875MHz level actually rises under load (it drives the bandwidth cap).

Usage:
  python3 benchmark/micro_peak_bench.py [--gpu 0]
"""
import argparse
import re
import subprocess
import threading
import time

import torch


# ── load-time clock sampling ─────────────────────────────────────────────
class ClockSampler:
    """Periodically snapshots sclk/mclk across all DCUs via rocm-smi."""

    def __init__(self):
        self._samples = []  # list of (list_of_sclk, list_of_mclk)
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None

    def _snapshot(self):
        try:
            out = subprocess.run(
                ["rocm-smi", "--showclocks"], capture_output=True,
                text=True, timeout=10).stdout
        except Exception:
            return None
        sclks, mclks = [], []
        for m in re.finditer(
                r"DCU\[\d+\]\s*:\s*(sclk|mclk) clock level: \d+ \((\d+)Mhz\)",
                out):
            if m.group(1) == "sclk":
                sclks.append(int(m.group(2)))
            else:
                mclks.append(int(m.group(2)))
        if not sclks and not mclks:
            return None
        return sclks, mclks

    def start(self):
        def loop():
            while not self._stop:
                s = self._snapshot()
                if s:
                    with self._lock:
                        self._samples.append(s)
                time.sleep(0.5)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def report(self, tag):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=15)
        with self._lock:
            ss = list(self._samples)
        if not ss:
            print(f"[clocks:{tag}] no samples captured")
            return
        s_all = [f for s in ss for f in s[0]]
        m_all = [f for s in ss for f in s[1]]
        def stat(v, name):
            return f"{name} min={min(v)} max={max(v)} avg={sum(v)//len(v)} MHz"
        print(f"[clocks:{tag}] n={len(ss)} samples | {stat(s_all, 'sclk')} | "
              f"{stat(m_all, 'mclk')}")


# ── micro benchmarks ─────────────────────────────────────────────────────
def _sustained(fn, secs):
    """Run fn() back-to-back for at least `secs` wall seconds so the DVFS
    controller has time to ramp sclk to the load level. Returns nothing."""
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    while True:
        fn()
        en.record()
        en.synchronize()
        if st.elapsed_time(en) >= secs * 1e3:
            break


def _run_timed(fn, secs):
    """Run fn() for ~`secs` seconds, return mean ms per call."""
    n = 0
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    while True:
        fn()
        n += 1
        en.record()
        en.synchronize()
        if st.elapsed_time(en) >= secs * 1e3:
            break
    return st.elapsed_time(en) / n


def bench_copy(nbytes, warmup_s=2.0, meas_s=2.0):
    n = nbytes // 4
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    _sustained(lambda: y.copy_(x), warmup_s)
    ms = _run_timed(lambda: y.copy_(x), meas_s)
    return 2 * nbytes / (ms / 1e3) / 1e9  # read + write, GB/s


def bench_read(nbytes, warmup_s=2.0, meas_s=2.0):
    n = nbytes // 4
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    _sustained(lambda: x.sum(), warmup_s)
    ms = _run_timed(lambda: x.sum(), meas_s)
    return nbytes / (ms / 1e3) / 1e9  # pure read, GB/s


def bench_gemm_fp16(M, N, K, warmup_s=8.0, meas_s=3.0):
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    _sustained(lambda: torch.matmul(a, b), warmup_s)
    ms = _run_timed(lambda: torch.matmul(a, b), meas_s)
    return 2.0 * M * N * K / (ms / 1e3) / 1e12  # TFLOPS


def bench_gemm_int8(M, N, K, warmup_s=8.0, meas_s=3.0):
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="cuda")
    has_intmm = hasattr(torch, "_int_mm")
    if has_intmm:
        try:
            torch._int_mm(a, b)
        except Exception as e:
            print(f"  torch._int_mm raised during probe: {e}")
            has_intmm = False
    if not has_intmm:
        print("  torch._int_mm unavailable; trying torch.matmul(int8,int8)...")
        try:
            torch.matmul(a, b)   # likely promotes to a higher dtype
            print("  WARNING: torch.matmul(int8) promoted dtype — NOT an "
                  "INT8-tensor-core measurement; result invalid for INT8 peak")
        except Exception as e:
            print(f"  torch.matmul(int8) also failed: {e}")
        return None
    _sustained(lambda: torch._int_mm(a, b), warmup_s)
    ms = _run_timed(lambda: torch._int_mm(a, b), meas_s)
    return 2.0 * M * N * K / (ms / 1e3) / 1e12  # TOPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--size-gb", type=float, default=2.0)
    ap.add_argument("--gemm", type=int, default=8192)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    dev = torch.device("cuda")
    print(f"torch {torch.__version__} | hip {torch.version.hip}")
    print(f"device {torch.cuda.get_device_name(args.gpu)} "
          f"({torch.cuda.get_device_capability(args.gpu)}) gpu={args.gpu}")

    sampler = ClockSampler()
    sampler.start()
    try:
        gb = int(args.size_gb * (1 << 30))
        cp = bench_copy(gb)
        rd = bench_read(gb)
        print(f"\nHBM bandwidth ({args.size_gb:.0f} GiB buffer):")
        print(f"  copy (read+write): {cp:.1f} GB/s")
        print(f"  read (sum reduce): {rd:.1f} GB/s")

        f16 = bench_gemm_fp16(args.gemm, args.gemm, args.gemm)
        print(f"\nFP16 GEMM {args.gemm}^3 peak: {f16:.1f} TFLOPS")

        i8 = bench_gemm_int8(args.gemm, args.gemm, args.gemm)
        if i8:
            print(f"INT8 GEMM {args.gemm}^3 peak: {i8:.1f} TOPS")
            print(f"  INT8/FP16 ratio: {i8 / f16:.2f}x")
    finally:
        sampler.report("load")

    print("\ndone")


if __name__ == "__main__":
    main()
