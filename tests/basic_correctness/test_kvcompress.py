"""Compare the outputs of HF and vLLM when using greedy sampling.

It tests KV-Compress with compression rate of 1.0 (no compression).
KV-Compress can be enabled with enable_kvcompress=True.

Run `pytest tests/models/test_chunked_prefill.py`.
"""
import pytest
from transformers import AutoTokenizer
from copy import deepcopy
import numpy as np

MODELS = [
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_no_compression(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        target_compression_rate=1.0,  # no compression
        protected_window_size=1,
        metric_collection_buffer_size=1,
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        print(hf_output_str)
        print(vllm_output_str)
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_all_protected(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        target_compression_rate=0.1,
        protected_window_size=1_000_000,
        metric_collection_buffer_size=1_000_000,
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        print(hf_output_str)
        print(vllm_output_str)
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("random_seed", [1])
@pytest.mark.parametrize("num_digits", [100])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_checkpoint_compression_schedule_without_bias(
    vllm_runner,
    random_digit_generator,
    checkpointer,
    random_seed: int,
    num_digits: int,
    model: str,
    dtype: str,
) -> None:
    """Checks for alignment between the vLLM integration and the experimental implementation
    that simulates eviction within a single forward pass.
    """
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        protected_window_size=50,
        metric_collection_buffer_size=10,
        save_checkpoint_dir='./checkpoints',
        # For consistency with experimental implementation
        max_cache_tokens=150,
        even_layer_evict=True,
        block_size=1,
    )
    random_digit_prompts, random_digit_responses = random_digit_generator(num_digits, random_seed)
    max_tokens = max(len(response) for response in random_digit_responses)
    tokenizer = AutoTokenizer.from_pretrained(model)
    reference_token_ids = [
        tokenizer.encode(completion, add_special_tokens=False)[1:]  # adds '' token
        for completion in random_digit_responses
    ]
    for i, (p, ref) in enumerate(zip(random_digit_prompts, reference_token_ids)):
        checkpointer.checkpoint(f'input_ids_{i}', tokenizer.encode(p))
        checkpointer.checkpoint(f'reference_token_ids_{i}', ref)

    topk_ll = 5
    vllm_model.generate_greedy_logprobs(random_digit_prompts,
                                        max_tokens,
                                        topk_ll,
                                        reference_token_ids=deepcopy(reference_token_ids))
    del vllm_model
    raise
