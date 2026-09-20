# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    compute_probs_and_sample_next_token,
)

DEVICE_TYPE = current_platform.device_type


def _seed_default_generator(seed: int) -> None:
    set_random_seed(seed)


def _make_sampling_metadata(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=DEVICE_TYPE),
        presence_penalties=torch.empty(0, device=DEVICE_TYPE),
        repetition_penalties=torch.empty(0, device=DEVICE_TYPE),
        output_token_ids=[[] for _ in range(batch_size)],
        spec_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_compute_probs_and_sample_next_token_uses_fp64_exponential_race():
    batch_size = 4
    vocab_size = 32
    generator = torch.Generator(device=DEVICE_TYPE).manual_seed(11)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
        generator=generator,
    )
    metadata = _make_sampling_metadata(batch_size)

    _seed_default_generator(12345)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty(probs.shape, dtype=torch.float64, device=probs.device)
    q.exponential_()
    expected_ids = q.reciprocal_().mul_(probs).argmax(dim=-1).view(-1)

    _seed_default_generator(12345)
    actual_ids, actual_probs = compute_probs_and_sample_next_token(
        logits.clone(),
        metadata,
        use_fp64_gumbel=True,
    )

    assert torch.equal(actual_ids, expected_ids)
    assert torch.allclose(actual_probs, probs)


@pytest.mark.parametrize(
    "top_k,top_p,weights",
    [
        (None, None, [0.5, 0.3, 0.15, 0.05]),
        (4, 1.0, [0.5, 0.3, 0.15, 0.05]),
        (2, None, [0.5, 0.3, 0.0, 0.0]),
        (None, 0.7, [0.5, 0.3, 0.0, 0.0]),
        (2, 0.6, [0.5, 0.0, 0.0, 0.0]),
    ],
)
def test_draft_sampling_returns_filtered_proposal_probs(top_k, top_p, weights):
    metadata = _make_sampling_metadata(4)
    metadata.temperature.fill_(2.0)
    if top_k is not None:
        metadata.top_k = torch.full((4,), top_k, device=DEVICE_TYPE)
    if top_p is not None:
        metadata.top_p = torch.full((4,), top_p, device=DEVICE_TYPE)
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05], device=DEVICE_TYPE).log() * 2
    token_ids, probs = compute_probs_and_sample_next_token(
        logits.expand(4, -1).clone(), metadata
    )
    expected = torch.tensor(weights, device=DEVICE_TYPE)
    expected /= expected.sum()
    torch.testing.assert_close(probs, expected.expand_as(probs))
    assert (probs.gather(1, token_ids[:, None]) > 0).all()


@pytest.mark.parametrize("all_greedy", [False, True])
def test_draft_sampling_preserves_greedy_argmax_with_ties(all_greedy):
    metadata = _make_sampling_metadata(2)
    metadata.all_greedy = all_greedy
    metadata.all_random = False
    metadata.temperature[0] = 0
    if all_greedy:
        metadata.temperature.zero_()
    metadata.top_p = torch.full((2,), 0.1, device=DEVICE_TYPE)
    logits = torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]], device=DEVICE_TYPE)
    token_ids, _ = compute_probs_and_sample_next_token(logits.clone(), metadata)
    assert torch.equal(token_ids, logits.argmax(dim=-1))


@pytest.mark.parametrize("num_drafts", [1, 3])
def test_parallel_drafts_use_each_requests_sampling_parameters(num_drafts):
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer._enable_probabilistic_draft_probs = True
    proposer.use_fp64_gumbel = False
    metadata = _make_sampling_metadata(2)
    metadata.temperature = torch.tensor([2.0, 1.0], device=DEVICE_TYPE)
    metadata.top_k = torch.tensor([1, 2], device=DEVICE_TYPE)
    metadata.top_p = torch.tensor([1.0, 0.9], device=DEVICE_TYPE)
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]], device=DEVICE_TYPE).log()
    logits = logits.expand(2 * num_drafts, -1).clone()
    token_ids, probs = proposer._sample_from_logits(logits, metadata)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.625, 0.375, 0.0, 0.0]], device=DEVICE_TYPE
    ).repeat_interleave(num_drafts, dim=0)
    torch.testing.assert_close(probs, expected)
    assert (probs.gather(1, token_ids[:, None]) > 0).all()
    assert metadata.top_k.shape == metadata.top_p.shape == (2,)
