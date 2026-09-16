from dataclasses import dataclass

import torch

from vllm.compilation.cuda_graph import (
    _collect_cudagraph_input_signatures,
    _format_cudagraph_signature_differences,
)


@dataclass
class Metadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class ForwardContext:
    attn_metadata: dict[str, Metadata]
    slot_mapping: dict[str, torch.Tensor]
    dp_metadata: object | None = None
    additional_kwargs: dict[str, torch.Tensor] | None = None


def _context(base: torch.Tensor) -> ForwardContext:
    return ForwardContext(
        attn_metadata={"layer": Metadata(base[:2], base[2:4])},
        slot_mapping={"layer": base[4:6]},
        additional_kwargs={"mask": base[6:8]},
    )


def _signatures(base: torch.Tensor, input_ids: torch.Tensor) -> dict:
    return _collect_cudagraph_input_signatures(
        (), {"input_ids": input_ids}, _context(base), include_cpu=True
    )


def test_cudagraph_signature_tracks_kwargs_and_forward_context() -> None:
    base = torch.arange(16)
    signatures = _signatures(base, base[:8])

    assert "kwargs['input_ids']" in signatures
    assert "forward_context.attn_metadata['layer'].query_start_loc" in signatures
    assert "forward_context.slot_mapping['layer']" in signatures
    assert "forward_context.additional_kwargs['mask']" in signatures


def test_cudagraph_signature_allows_new_views_of_same_storage() -> None:
    base = torch.arange(16)
    capture = _signatures(base, base[:8])
    replay = _signatures(base, base[:8])

    assert _format_cudagraph_signature_differences(capture, replay) is None


def test_cudagraph_signature_reports_changed_storage_and_view() -> None:
    base = torch.arange(16)
    capture = _signatures(base, base[:8])
    cloned = _signatures(base, base[:8].clone())
    shifted = _signatures(base, base[1:9])

    assert "kwargs['input_ids']" in _format_cudagraph_signature_differences(
        capture, cloned
    )
    assert "kwargs['input_ids']" in _format_cudagraph_signature_differences(
        capture, shifted
    )




def test_cudagraph_signature_ignores_cpu_tensors_by_default() -> None:
    cpu = torch.arange(8)
    context = ForwardContext(
        attn_metadata={"layer": Metadata(cpu[:2], cpu[2:4])},
        slot_mapping={"layer": cpu[4:6]},
    )

    assert _collect_cudagraph_input_signatures((), {"input_ids": cpu}, context) == {}


def test_piecewise_signature_ignores_attention_context() -> None:
    base = torch.arange(16)
    signatures = _collect_cudagraph_input_signatures(
        (),
        {"input_ids": base[:8]},
        _context(base),
        include_cpu=True,
        include_attention_context=False,
    )

    assert "kwargs['input_ids']" in signatures
    assert not any("attn_metadata" in path for path in signatures)
    assert not any("slot_mapping" in path for path in signatures)
