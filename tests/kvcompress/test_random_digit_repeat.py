import pytest
from transformers import AutoTokenizer
from copy import deepcopy
import numpy as np
import torch

from vllm.debug import RandomDigitCheckpointConfig

MODELS = [
    # "NousResearch/Llama-2-7b-hf",
    # "NousResearch/Hermes-2-Theta-Llama-3-8B",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

MODEL_KV_HEAD_BIAS = {
    "NousResearch/Llama-2-7b-hf": "./kv_head_bias.npz",
    "NousResearch/Hermes-2-Theta-Llama-3-8B": "./kv_head_bias_llama3.npz",
    "mistralai/Mistral-7B-Instruct-v0.2": "./kv_head_bias_mistral.npz",
}


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
    vllm_outputs = vllm_model.generate_greedy_logprobs(
                                                max_tokens,
                                                topk_ll,
                                                prompts=random_digit_prompts,
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


@pytest.mark.parametrize("random_seed", [1])
@pytest.mark.parametrize("num_digits", [100])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_parity_with_simulated_compression(
    vllm_runner,
    tokenizer_dependent_random_digit_generator,
    checkpointer,
    random_seed: int,
    num_digits: int,
    model: str,
    dtype: str,
) -> None:
    """Checks for alignment between the vLLM integration and the experimental implementation
    that simulates eviction within a single forward pass.
    """
    checkpoint_cfg = RandomDigitCheckpointConfig(
        model_name=model,
        max_cache_tokens=64,
        protected_window_size=32,
        metric_collection_buffer_size=10,
        num_digits=num_digits,
        control_layers=[0, 1],
    )

    # Don't checkpoint during profiling
    checkpointer.set_condition(checkpoint_layer=lambda _: False)
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        # max_cache_tokens=checkpoint_cfg.max_cache_tokens,
        # target_compression_rate=0.1,
        block_size=16,
        # protected_window_size=checkpoint_cfg.protected_window_size,
        # metric_collection_buffer_size=checkpoint_cfg.metric_collection_buffer_size,
        even_layer_evict=False, #True,
        control_layers=[], #checkpoint_cfg.control_layers,
        # save_checkpoint_dir='./checkpoint',
        kv_head_bias_path=MODEL_KV_HEAD_BIAS[model],
        kv_head_bias_weight=50000,
        new_token_limit=100,
        compression_interval=1000,
    )
    checkpointer.set_config(checkpoint_cfg)

    # Only checkpoint the first non-controlled layer
    checkpointer.set_condition(checkpoint_layer=lambda layer: layer == 2)

    tokenizer = AutoTokenizer.from_pretrained(model)
    input_token_ids, reference_token_ids = tokenizer_dependent_random_digit_generator(
        tokenizer, num_digits, random_seed, n_seqs=100, repeat_len=10)

    if "Llama-2" in model or "Mistral" in model:
        # Llama-2 tokenizer adds an empty "" token before digits
        reference_token_ids = [ref_tokens[1:] for ref_tokens in reference_token_ids]

    # total_tokens = [
    #     len(tokenizer.encode(prompt)) + len(completion_token_ids)
    #     for prompt, completion_token_ids in zip(random_digit_prompts, reference_token_ids)
    # ]
    max_tokens = max(len(response) for response in reference_token_ids)

    checkpointer.checkpoint('input_token_ids', torch.tensor(input_token_ids[0]))
    checkpointer.checkpoint('reference_token_ids', torch.tensor(reference_token_ids[0]))

    print(tokenizer.decode(input_token_ids[0]))
    print(tokenizer.decode(reference_token_ids[0]))

    topk_ll = 0
    vllm_outputs = vllm_model.generate_greedy_logprobs(
        max_tokens,
        topk_ll,
        prompt_token_ids=input_token_ids,
        reference_token_ids=deepcopy(reference_token_ids),
        max_cache_tokens=checkpoint_cfg.max_cache_tokens,
        # target_compression_rate=0.1,
        protected_window_size=checkpoint_cfg.protected_window_size,
        metric_collection_buffer_size=checkpoint_cfg.metric_collection_buffer_size,
    )
    del vllm_model
    print(f"TOTAL TOKENS = {sum([len(i) + len(r) for i, r in zip(input_token_ids, reference_token_ids)])}")

    def is_correct(logprobs, index):
        max_i = None
        max_logprob = -float('inf')
        for i in logprobs:
            logprob = logprobs[i].logprob
            if logprob > max_logprob:
                max_logprob = logprob
                max_i = i
        return max_i == index

    for i, (ref_token_ids, (_, output, logprobs)) in enumerate(
        zip(reference_token_ids, vllm_outputs)
    ):
        assert len(logprobs) == len(ref_token_ids)
        for i, (d, t) in enumerate(zip(logprobs, ref_token_ids)):
            if t not in d:
                raise Exception(f"{i}/{len(logprobs)} fail ({t} not in {d})")
        nll = [-d[t].logprob for d, t in zip(logprobs, ref_token_ids)]
        ppl = float("%.4f" % np.exp(sum(nll) / len(nll)))
        acc = float("%.2f" % (sum([is_correct(d, t) for d, t in zip(logprobs, ref_token_ids)]) / len(logprobs) * 100.))
        print(f"{acc=}, {ppl=}")
        assert ppl < 1.3
        assert acc > 95
        # vllm_output = output[:len(reference_completion)]
        # assert reference_completion == vllm_output, (
        #     f"Test{i}:\nReference: {reference_completion!r}\nvLLM: {vllm_output!r}")


@pytest.mark.parametrize("random_seed", [1])
@pytest.mark.parametrize("num_digits", [100])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_compression_without_bias(
    vllm_runner,
    random_digit_generator,
    random_seed: int,
    num_digits: int,
    model: str,
    dtype: str,
) -> None:
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        target_compression_rate=0.8,
        protected_window_size=100,
        metric_collection_buffer_size=10,
    )
    random_digit_prompts, random_digit_responses = random_digit_generator(num_digits, random_seed)
    max_tokens = max(len(response) for response in random_digit_responses)
    tokenizer = AutoTokenizer.from_pretrained(model)
    reference_token_ids = [
        tokenizer.encode(completion, add_special_tokens=False)[1:]  # adds '' token
        for completion in random_digit_responses
    ]
    total_tokens = [
        len(tokenizer.encode(prompt)) + len(completion_token_ids)
        for prompt, completion_token_ids in zip(random_digit_prompts, reference_token_ids)
    ]
    topk_ll = 5
    vllm_outputs = vllm_model.generate_greedy_logprobs(
                                                max_tokens,
                                                topk_ll,
                                                prompts=random_digit_prompts,
                                                reference_token_ids=deepcopy(reference_token_ids))
    del vllm_model
    print(f"TOTAL_KVS: {total_tokens} * 1024 = {[t*1024 for t in total_tokens]}")

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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_compression_with_bias(
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
        target_compression_rate=0.8,
        protected_window_size=100,
        metric_collection_buffer_size=10,
        kv_head_bias_path="/irehg/vllm/kv_head_bias.npz",
    )
    max_tokens = max(len(response) for response in random_digit_responses)
    tokenizer = AutoTokenizer.from_pretrained(model)
    reference_token_ids = [
        tokenizer.encode(completion, add_special_tokens=False)[1:]  # adds '' token
        for completion in random_digit_responses
    ]
    total_tokens = [
        len(tokenizer.encode(prompt)) + len(completion_token_ids)
        for prompt, completion_token_ids in zip(random_digit_prompts, reference_token_ids)
    ]
    topk_ll = 5
    vllm_outputs = vllm_model.generate_greedy_logprobs(
                                                max_tokens,
                                                topk_ll,
                                                prompts=random_digit_prompts,
                                                reference_token_ids=deepcopy(reference_token_ids))
    del vllm_model
    print(f"TOTAL_KVS: {total_tokens} * 1024 = {[t*1024 for t in total_tokens]}")

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

