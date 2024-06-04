import pytest
from transformers import AutoTokenizer
from copy import deepcopy
import numpy as np

MODELS = [
    "NousResearch/Llama-2-7b-hf",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_no_compression(
    vllm_runner,
    random_digit_prompts,
    random_digit_responses,
    model: str,
    dtype: str,
) -> None:
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        target_compression_rate=1.0,
        protected_window_size=32,
        metric_collection_buffer_size=16,
    )
    max_tokens = max(len(response) for response in random_digit_responses)
    tokenizer = AutoTokenizer.from_pretrained(model)
    reference_token_ids = [
        tokenizer.encode(completion, add_special_tokens=False)[1:]  # adds '' token
        for completion in random_digit_responses
    ]
    topk_ll = 5
    vllm_outputs = vllm_model.generate_greedy_logprobs(random_digit_prompts,
                                                max_tokens,
                                                topk_ll,
                                                reference_token_ids=deepcopy(reference_token_ids))
    del vllm_model

    for i, (reference_completion, ref_token_ids, (_, output, logprobs)) in enumerate(
        zip(random_digit_responses, reference_token_ids, vllm_outputs)
    ):
        ppl = np.exp(sum([-d[t].logprob for d, t in zip(logprobs, ref_token_ids)]) / len(logprobs))
        assert ppl < 1.01
        vllm_output = output[:len(reference_completion)]
        assert reference_completion == vllm_output, (
            f"Test{i}:\nReference: {reference_completion!r}\nvLLM: {vllm_output!r}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_compression_without_bias(
    vllm_runner,
    random_digit_prompts,
    random_digit_responses,
    model: str,
    dtype: str,
) -> None:
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        target_compression_rate=0.9,
        protected_window_size=32,
        metric_collection_buffer_size=16,
    )
    max_tokens = max(len(response) for response in random_digit_responses)
    tokenizer = AutoTokenizer.from_pretrained(model)
    reference_token_ids = [
        tokenizer.encode(completion, add_special_tokens=False)[1:]  # adds '' token
        for completion in random_digit_responses
    ]
    topk_ll = 5
    vllm_outputs = vllm_model.generate_greedy_logprobs(random_digit_prompts,
                                                max_tokens,
                                                topk_ll,
                                                reference_token_ids=deepcopy(reference_token_ids))
    del vllm_model

    for i, (reference_completion, ref_token_ids, (_, output, logprobs)) in enumerate(
        zip(random_digit_responses, reference_token_ids, vllm_outputs)
    ):
        nll = [-d[t].logprob for d, t in zip(logprobs, ref_token_ids) if t in d]
        ppl = np.exp(sum(nll) / len(nll))
        print(ppl)
        assert ppl < 1.01
        vllm_output = output[:len(reference_completion)]
        assert reference_completion == vllm_output, (
            f"Test{i}:\nReference: {reference_completion!r}\nvLLM: {vllm_output!r}")
