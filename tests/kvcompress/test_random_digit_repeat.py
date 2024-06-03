import pytest

MODELS = [
    "NousResearch/Llama-2-7b-hf",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_random_digit_repeat_no_compression(
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
    topk_ll = 10
    vllm_outputs = vllm_model.generate_greedy_logprobs(random_digit_prompts,
                                                       max_tokens,
                                                       topk_ll,
                                                       random_digit_responses)
    del vllm_model

    for i, (reference_completion, prompt, (_, output)) in enumerate(
        zip(random_digit_responses, random_digit_prompts, vllm_outputs)
    ):
        print(output)
        reference = prompt + reference_completion
        vllm_output = output[:len(reference)]
        assert reference == vllm_output, (
            f"Test{i}:\nReference: {reference!r}\nvLLM: {vllm_output!r}")
