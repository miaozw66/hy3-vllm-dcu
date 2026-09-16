#!/usr/bin/env python3
"""vLLM W8A8 INT8 GEMM -> custom HIP kernels (TP8, M=16 / M=4096) integration.

Importing this module ONLY registers an import hook (no CUDA, no vLLM import),
so it is safe to import before vLLM spawns its EngineCore workers.

At vLLM module load time it patches:
  - TritonInt8ScaledMMLinearKernel / CutlassInt8ScaledMMLinearKernel
      process_weights_after_loading : after original weight prep, pack the
        [K,N] weight into the layout expected by the HIP kernels (once).
      apply_weights : for exact (m,n,k) in the routed tables, run the HIP
        kernel and return its output (use mode); validate mode also computes
        the original result, logs the diff, and returns the original result.

Routed shapes: 7 of the 8 TP8 shapes (all measured FASTER than the Triton
baseline on 2026-08-27). shared_down_proj_m4096 (4096,4096,192) measured
2.38x SLOWER -> deliberately NOT routed; it falls through to the original
Triton implementation.
"""
import os
import torch  # noqa: F401  (no CUDA init at import; needed in patch closures)
from importlib.abc import Loader, MetaPathFinder

_LOG_PATH = "/tmp/zth_w8a8_diffs.txt"
_TARGET_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.triton"

# (m, n, k) -> extension namespace suffix
M16_SHAPES = {
    (16, 4096, 1024): "o_proj_m16",
    (16, 1280, 4096): "qkv_proj_m16",
    (16, 4096, 192): "shared_down_proj_m16",
    (16, 384, 4096): "shared_gate_up_proj_m16",
}
M4096_SHAPES = {
    (4096, 4096, 1024): "o_proj_m4096",
    (4096, 1280, 4096): "qkv_proj_m4096",
    (4096, 384, 4096): "shared_gate_up_proj_m4096",
    # NOTE: (4096, 4096, 192) shared_down_proj_m4096 measured 2.38x SLOWER
    # than triton (0.785ms vs 0.330ms) -> intentionally NOT routed.
}
# (k, n) -> pack module per M family (each .hip is self-consistent: its own
# pack layout is what its gemm consumes; M16 and M4096 packs may differ, e.g.
# shared_gate_up_proj).
KN_PACK_M16 = {
    (4096, 1280): "qkv_proj_m16",
    (1024, 4096): "o_proj_m16",
    (4096, 384): "shared_gate_up_proj_m16",
    (192, 4096): "shared_down_proj_m16",
}
KN_PACK_M4096 = {
    (4096, 1280): "qkv_proj_m4096",
    (1024, 4096): "o_proj_m4096",
    (4096, 384): "shared_gate_up_proj_m4096",
    # (192, 4096): "shared_down_proj_m4096"  # not routed -> no M4096 pack
}
# m <= 32 small-M path. Each .hip's launch_w8a8_gemm now has an m <= 32
# split-k DUMMA arm consuming the SAME packed n-major [N, K] layout as its
# M4096 arm, so the values reuse the m4096 extension names. The 3 shared
# shapes reuse _zth_packed4096 (identical pack); (192, 4096) has no M4096
# pack (down m>=4096 is intentionally unrouted), so down is packed into its
# own _zth_packed32 via shared_down_proj_m4096.pack_weight.
KN_PACK_M32 = {
    (4096, 1280): "qkv_proj_m4096",
    (1024, 4096): "o_proj_m4096",
    (4096, 384): "shared_gate_up_proj_m4096",
    (192, 4096): "shared_down_proj_m4096",
}
# (k, n) -> route index. process() stamps layer._zth_route so the dispatch op
# can pick the right gemm_out namespace (m16 .so vs m4096 .so) without dynamo
# ever seeing an int/tuple attribute.
_ROUTE_BY_KN = {
    (4096, 1280): 0,  # qkv_proj
    (1024, 4096): 1,  # o_proj
    (4096, 384): 2,   # shared_gate_up_proj
    (192, 4096): 3,   # shared_down_proj
}
# route -> (m16 ext name, m4096 ext name). Both m <= 32 (split-k arm) and
# m >= 4096 (main arm) live in the m4096 .so; only m == 16 uses the m16 .so.
_ROUTE_NAMES = [
    ("qkv_proj_m16", "qkv_proj_m4096"),
    ("o_proj_m16", "o_proj_m4096"),
    ("shared_gate_up_proj_m16", "shared_gate_up_proj_m4096"),
    ("shared_down_proj_m16", "shared_down_proj_m4096"),
]

_state = {"modules": None, "ops": None, "patched": set()}
# Must keep the Library object alive for the whole process: torch 2.10 drops
# the schema when the Library is GC'd, but leaves register_fake/impl behind,
# producing "Could not find schema ... did you forget to def() the operator?".
_DISPATCH_LIB = None
_MVAL_COUNT = {"n": 0}
_MODE = os.environ.get("ZTH_W8A8_MODE", "use")  # validate | use | off
_DBG = os.environ.get("ZTH_W8A8_DEBUG", "0") == "1"
_DBG_PATH = "/tmp/zth_apply_debug.txt"


def _dbg(*parts):
    """Lightweight routing debug. During CUDA-graph capture this writes one
    line per call (host I/O does not disturb graph capture); never in replay."""
    if not _DBG:
        return
    if torch.compiler.is_compiling():
        # -O1 (VLLM_COMPILE) dynamo 会逐行内联追踪 apply_weights；open() 写文件
        # 会触发 `Failed to trace builtin operator` graph break。编译期间跳过写
        # 文件（eager 运行时仍记录），避免破坏编译。见 HY3_DCU 优化记录。
        return
    try:
        with open(_DBG_PATH, "a") as f:
            f.write(f"{os.getpid()}\t" + "\t".join(str(p) for p in parts) + "\n")
    except Exception:
        pass


def _init():
    if _state["modules"] is not None:
        return
    import torch
    from vllm import _custom_ops as ops
    import zth_w8a8_ext
    failed = zth_w8a8_ext.load_all()
    if failed:
        raise RuntimeError(f"zth_w8a8 extensions failed to load: {failed}")
    mods = {}
    for name in set(list(M16_SHAPES.values()) + list(M4096_SHAPES.values())
                    + list(KN_PACK_M16.values()) + list(KN_PACK_M4096.values())
                    + list(KN_PACK_M32.values())):
        mods[name] = getattr(torch.ops, f"zth_{name}")
    _state["modules"] = mods
    _state["ops"] = ops


def _triton_fallback(x, w_q, w_s, bias):
    """Original Triton path (what apply_weights did before this patch).

    Only reached for shapes/conditions we deliberately do not route: down
    m >= 4096, m in 33..4095, un-packable layers, or bias is not None.
    Runs inside the dispatch op's impl, i.e. outside dynamo tracing.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )
    ops = _state["ops"]
    x_q, x_s, x_zp = ops.scaled_int8_quant(
        x.contiguous(), None, None, symmetric=True)
    return triton_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s,
                            out_dtype=x.dtype, bias=bias)


def _workspace_bytes(m: int, n: int, k: int) -> int:
    bytes_per_partial = m * n * 4
    budget = 16 * 1024 * 1024
    cap = min(16, budget // bytes_per_partial, k // 32)
    return max(256, cap * bytes_per_partial)


def _log(shape, md, mean, gt):
    with open(_LOG_PATH, "a") as f:
        f.write(f"{os.getpid()}\t{shape}\t{md:.8f}\t{mean:.8f}\t{gt}\n")


def _make_patch(cls):
    if cls in _state["patched"]:
        return
    orig_apply = cls.apply_weights
    orig_process = cls.process_weights_after_loading

    def process(self, layer):
        orig_process(self, layer)
        try:
            w = layer.weight
            if w is None or w.ndim != 2:
                return
            k, n = w.shape
            p16 = KN_PACK_M16.get((k, n))
            p4096 = KN_PACK_M4096.get((k, n))
            p32 = KN_PACK_M32.get((k, n))
            if p16 is None and p4096 is None and p32 is None:
                return
            route = _ROUTE_BY_KN.get((k, n))
            if route is None:
                return
            layer._zth_route = route
            _init()
            ws = layer.weight_scale
            if ws is None:
                return
            ws1 = ws.reshape(n, 1) if ws.ndim == 1 else ws
            if tuple(ws1.shape) != (n, 1):
                return
            wc = w.detach().contiguous()
            ws1c = ws1.detach().contiguous()
            mods = _state["modules"]
            if p16 is not None:
                pk16, ps16 = mods[p16].pack_weight(wc, ws1c)
                layer._zth_packed16 = pk16
                layer._zth_packed_scale16 = ps16
                layer._zth_ws16 = torch.empty(
                    _workspace_bytes(16, n, k), dtype=torch.uint8,
                    device=w.device)
            if p4096 is not None:
                pk4096, ps4096 = mods[p4096].pack_weight(wc, ws1c)
                layer._zth_packed4096 = pk4096
                layer._zth_packed_scale4096 = ps4096
                layer._zth_ws4096 = torch.empty(
                    _workspace_bytes(4096, n, k), dtype=torch.uint8,
                    device=w.device)
            if p32 is not None:
                if p4096 is not None and p32 == p4096:
                    # Same .so / same n-major pack: reuse the M4096 buffer.
                    layer._zth_packed32 = layer._zth_packed4096
                    layer._zth_packed_scale32 = layer._zth_packed_scale4096
                else:
                    pk32, ps32 = mods[p32].pack_weight(wc, ws1c)
                    layer._zth_packed32 = pk32
                    layer._zth_packed_scale32 = ps32
                layer._zth_ws32 = torch.empty(
                    _workspace_bytes(32, n, k), dtype=torch.uint8,
                    device=w.device)
            _dbg("PROCESS", f"kn={k},{n}", f"p16={p16}", f"p4096={p4096}",
                 f"p32={p32}",
                 f"pk16={getattr(layer,'_zth_packed16',None) is not None}",
                 f"pk4096={getattr(layer,'_zth_packed4096',None) is not None}",
                 f"pk32={getattr(layer,'_zth_packed32',None) is not None}",
                 f"ws32={getattr(layer,'_zth_ws32',None) is not None}")
        except Exception as exc:  # never break model loading
            with open(_LOG_PATH, "a") as f:
                f.write(f"{os.getpid()}\tPACK_ERR\t{exc}\n")

    def apply(self, layer, x, bias=None):
        if _MODE == "off":  # pure original path, zero overhead (A/B baseline)
            return orig_apply(self, layer, x, bias)
        # Whole routing delegates to the black-box torch custom op below.
        # Under -O1 (VLLM_COMPILE) dynamo inline-traces this method: any Python
        # control flow / attribute read here (m branch, _zth_kn int tuple, _dbg
        # file I/O) leaks non-tensor values into the compiled graph and crashes
        # (ConstraintViolation, "Expected tensors only, got int", open() graph
        # break). The op hides all of that from dynamo, exactly like FusedMoE's
        # moe_forward: dynamo sees one op call + a register_fake shape rule.
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
        return torch.ops.zth_dispatch.zth_scaled_mm_dispatch(
            x, w_q, w_s, bias,
            getattr(layer, "_zth_packed16", None),
            getattr(layer, "_zth_packed_scale16", None),
            getattr(layer, "_zth_ws16", None),
            getattr(layer, "_zth_packed32", None),
            getattr(layer, "_zth_packed_scale32", None),
            getattr(layer, "_zth_ws32", None),
            getattr(layer, "_zth_ws4096", None),
            getattr(layer, "_zth_route", -1),
        )

    cls.apply_weights = apply
    cls.process_weights_after_loading = process
    _state["patched"].add(cls)


def _patch(module):
    try:
        from vllm.model_executor.kernels.linear.scaled_mm import cutlass
        _make_patch(cutlass.CutlassInt8ScaledMMLinearKernel)
    except Exception:
        pass
    try:
        _make_patch(module.TritonInt8ScaledMMLinearKernel)
    except Exception as exc:
        with open(_LOG_PATH, "a") as f:
            f.write(f"{os.getpid()}\tPATCH_ERR\t{exc}\n")


class _Finder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None:
                break
        if spec is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


class _Loader(Loader):
    def __init__(self, orig_loader):
        self._orig_loader = orig_loader

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        _patch(module)


import sys  # noqa: E402

sys.meta_path.insert(0, _Finder())


def _register_dispatch_op():
    """Register the scaled_mm routing as one black-box torch custom op.

    Under -O1 (VLLM_COMPILE) dynamo inline-traces our patched apply_weights.
    Any Python control flow / attribute read in it (m branch, _zth_kn int
    tuple, _dbg file I/O) leaks non-tensor values into the compiled graph and
    crashes (ConstraintViolation, "Expected tensors only, got int", open()
    graph break). Moving all routing into a registered op's CUDA impl hides it
    from dynamo exactly like FusedMoE's moe_forward: dynamo sees one op call +
    a register_fake shape rule, and the runtime executes the real HIP kernels.
    """
    try:
        from torch.library import Library, impl, register_fake
    except ImportError:
        from torch.library import Library, impl
        from torch.library import impl_abstract as register_fake

    global _DISPATCH_LIB
    if _DISPATCH_LIB is not None:
        return  # already registered (re-import in the same process)
    _DISPATCH_LIB = Library("zth_dispatch", "DEF")
    try:
        _DISPATCH_LIB.define(
            "zth_scaled_mm_dispatch(Tensor x, Tensor w_q, Tensor w_s, "
            "Tensor? bias, "
            "Tensor? packed16, Tensor? pscale16, Tensor? ws16, "
            "Tensor? packed_mid, Tensor? pscale_mid, "
            "Tensor? ws_m32, Tensor? ws_m4096, int route) -> Tensor"
        )
    except Exception as exc:
        print(f"[zth_w8a8] dispatch op define skipped: {exc}")
        return  # already defined (re-import in the same process)

    def _fake(x, w_q, w_s, bias, p16, ps16, ws16, pmid, psmid, ws32, ws4096,
              route):
        return torch.empty((x.size(0), w_s.size(0)), dtype=torch.bfloat16,
                           device=x.device)

    register_fake("zth_dispatch::zth_scaled_mm_dispatch")(_fake)

    @impl("zth_dispatch::zth_scaled_mm_dispatch", "CUDA")
    def _dispatch(x, w_q, w_s, bias, p16, ps16, ws16, pmid, psmid, ws32,
                  ws4096, route):
        # ---- runtime routing: never traced by dynamo ----
        _init()
        if route < 0 or bias is not None:
            return _triton_fallback(x, w_q, w_s, bias)
        m = x.shape[-2]
        if m == 16:
            packed, pscale, ws, name = p16, ps16, ws16, _ROUTE_NAMES[route][0]
        elif m <= 32 and m != 16:
            # split-k arm lives in the m4096 .so, consumes the n-major pack
            packed, pscale, ws, name = (pmid, psmid, ws32,
                                        _ROUTE_NAMES[route][1])
        elif m >= 4096:
            packed, pscale, ws, name = (pmid, psmid, ws4096,
                                        _ROUTE_NAMES[route][1])
        else:  # m in 33..4095 -> no HIP arm, back to Triton
            return _triton_fallback(x, w_q, w_s, bias)
        if packed is None or ws is None:
            return _triton_fallback(x, w_q, w_s, bias)
        if _DBG:
            _dbg("DISPATCH", f"route={route}", f"m={m}",
                 f"kn={x.shape[-1]},{w_s.size(0)}", f"name={name}",
                 f"bias={bias is not None}")
        ops = _state["ops"]
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True)
        mod = _state["modules"][name]
        out = mod.gemm_out(x_q, packed, x_s, pscale, ws)
        if _MODE == "use":
            return out
        y_ref = _triton_fallback(x, w_q, w_s, bias)
        d = (out.float() - y_ref.float()).abs()
        _log((m, x.shape[-1], w_s.size(0)), d.max().item(), d.mean().item(),
             (d > 1.0).sum().item())
        return y_ref


_register_dispatch_op()


def report():
    """Aggregate the per-call diff log (call from the main process)."""
    if not os.path.exists(_LOG_PATH):
        print("no diff log")
        return
    rows = []
    with open(_LOG_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5 or parts[1] in ("PACK_ERR", "PATCH_ERR"):
                continue
            try:
                rows.append((parts[0], parts[1], float(parts[2]),
                             float(parts[3]), int(parts[4])))
            except ValueError:
                continue
    if not rows:
        print("no valid diff rows")
        return
    by_shape = {}
    for pid, shape, md, mean, gt in rows:
        by_shape.setdefault(shape, []).append((md, mean, gt))
    print(f"total calls: {len(rows)}")
    for shape, items in sorted(by_shape.items()):
        mds = [i[0] for i in items]
        gts = sum(i[2] for i in items)
        print(f"shape {shape}: calls={len(items)} max_diff={max(mds):.8f} "
              f"total_gt1={gts}")
