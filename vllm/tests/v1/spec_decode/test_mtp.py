# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.models.hy_v3_mtp import HYV3MultiTokenPredictorLayer
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.eagle import EagleProposer

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"


def _create_mtp_proposer(num_speculative_tokens: int) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=current_platform.device_type),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    return EagleProposer(vllm_config=vllm_config, device=current_platform.device_type)


def test_hyv3_mtp_uses_raw_residual_before_final_norm():
    mlp_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    residual = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    mtp_block = mock.MagicMock(return_value=(mlp_output, residual))
    final_layernorm = mock.MagicMock(side_effect=lambda value: value)
    layer = SimpleNamespace(
        enorm=lambda value: value,
        hnorm=lambda value: value,
        eh_proj=lambda value: value[:, :2],
        mtp_block=mtp_block,
        final_layernorm=final_layernorm,
    )

    result = HYV3MultiTokenPredictorLayer.forward(
        layer,
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([1, 2]),
        previous_hidden_states=torch.zeros(2, 2),
        inputs_embeds=torch.ones(2, 2),
    )

    expected = residual + mlp_output
    assert torch.equal(result, expected)
    final_layernorm.assert_called_once()
    assert torch.equal(final_layernorm.call_args.args[0], expected)


def test_mtp_single_token_returns_before_multi_token_path():
    batch_size = 2
    num_tokens = 4
    sample_indices = torch.tensor([1, 3])
    last_hidden_states = torch.arange(num_tokens * 2, dtype=torch.float32).view(
        num_tokens, 2
    )
    common_attn_metadata = SimpleNamespace(
        batch_size=lambda: batch_size,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )
    metadata_builder = mock.MagicMock()
    metadata_builder.build_for_drafting.return_value = object()
    attn_group = SimpleNamespace(
        get_metadata_builder=lambda: metadata_builder,
        layer_names=["mtp_layer"],
    )
    determine_batch = mock.MagicMock(return_value=(None, num_tokens, num_tokens))
    greedy_sample = mock.MagicMock(return_value=torch.tensor([41, 42]))
    model = mock.MagicMock(return_value=last_hidden_states)
    proposer = SimpleNamespace(
        method="mtp",
        set_inputs_first_pass=lambda **_: (
            num_tokens,
            sample_indices,
            common_attn_metadata,
        ),
        runner=object(),
        draft_attn_groups=[attn_group],
        _determine_batch_execution_and_padding=determine_batch,
        supports_mm_inputs=False,
        input_ids=torch.arange(num_tokens),
        _get_positions=lambda count: torch.arange(count),
        pass_hidden_states_to_model=False,
        vllm_config=object(),
        _get_slot_mapping=lambda *_: {},
        model=model,
        model_returns_tuple=lambda: False,
        num_speculative_tokens=1,
        parallel_drafting=False,
        _greedy_sample=greedy_sample,
        uses_mrope=False,
        positions=torch.arange(num_tokens),
        allowed_attn_types=None,
        arange=torch.arange(batch_size + 1, dtype=torch.int32),
        token_arange_np=torch.arange(batch_size + 1, dtype=torch.int32).numpy(),
        block_size=16,
    )

    with mock.patch(
        "vllm.v1.spec_decode.eagle.set_forward_context", return_value=nullcontext()
    ):
        result = EagleProposer.propose(
            proposer,
            target_token_ids=torch.tensor([10, 11, 12, 13]),
            target_positions=torch.arange(num_tokens),
            target_hidden_states=last_hidden_states,
            next_token_ids=torch.tensor([20, 21]),
            token_indices_to_sample=None,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=mock.MagicMock(),
        )

    assert torch.equal(result, torch.tensor([[41], [42]]))
    model.assert_called_once()
    determine_batch.assert_called_once_with(num_tokens)
    greedy_sample.assert_called_once()
    assert torch.equal(
        greedy_sample.call_args.args[0], last_hidden_states[sample_indices]
    )


@mock.patch("vllm.v1.spec_decode.eagle.get_pp_group")
@mock.patch("vllm.v1.spec_decode.eagle.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.eagle.get_model")
def test_mtp_load_model_unified(mock_get_model, mock_get_layers, mock_get_pp_group):
    """Test MTP-specific model loading with unified model approach."""

    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_attn_1": mock.MagicMock()}
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens


@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly"""

    device = torch.device(current_platform.device_type)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)
