"""sitecustomize: auto-register the zth_w8a8 vLLM integration hook in every
python process started with PYTHONPATH=/home/hy3-TP8/entry-vllm (including
vLLM spawned EngineCore workers). Never breaks interpreter startup."""
try:
    import zth_w8a8_integrate  # noqa: F401  (registers import hook only)
except Exception:  # noqa: BLE001
    pass

try:
    import zth_routed_moe_integrate  # noqa: F401  (optional routed-MoE hook)
except Exception:  # noqa: BLE001
    pass
