import pytest
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
from copy import deepcopy

from vllm.debug import RandomDigitCheckpointConfig
from vllm.sampling_params import SamplingParams

MODELS = [
    # "NousResearch/Llama-2-7b-hf",
    # "NousResearch/Hermes-2-Theta-Llama-3-8B",
    # "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-3.1-8B-Instruct",
]

MODEL_KV_HEAD_BIAS = {
    "NousResearch/Llama-2-7b-hf": "./kv_head_bias.npz",
    "NousResearch/Hermes-2-Theta-Llama-3-8B": "./kv_head_bias_llama3.npz",
    "mistralai/Mistral-7B-Instruct-v0.2": "./kv_head_bias_mistral.npz",
    "meta-llama/Llama-3.1-8B-Instruct": None,
}

NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day"
QUESTION = "What is the best thing to do in San Francisco?"
ANSWER_PREFIX = "The best thing to do in San Francisco is"
CORRECT_ANSWER = "eat a sandwich and sit in Dolores Park on a sunny day"

CHECK_LAYERS = [0, 1]

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("chunk_size", [1008])  # -1 1008
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_chunked_compression_max_batch(
    vllm_runner,
    checkpointer,
    model: str,
    dtype: str,
    chunk_size: int,
) -> None:

    # Don't checkpoint during profiling
    max_model_len = 10000
    checkpointer.set_condition(checkpoint_layer=lambda _: False)
    checkpointer.do_save = False
    checkpointer.do_validate = False
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        enable_chunked_prefill=True,
        block_size=16,
        even_layer_evict=False,
        control_layers=[],
        kv_head_bias_path=None,
        kv_head_bias_weight=50000,
        new_token_limit=100,
        compression_interval=1000,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=chunk_size if chunk_size > 0 else max_model_len + 500,
        max_num_seqs=min(256, chunk_size) if chunk_size > 0 else 256,
        max_model_len=max_model_len + 500,
    )
    checkpointer.do_save = False
    checkpointer.do_validate = True
    checkpointer.override_output = True
    checkpointer.base_dir = './checkpoints'
    checkpointer.tolerance = 1e-2
    checkpointer.set_condition(checkpoint_layer=lambda x: x in CHECK_LAYERS)
    checkpointer.enable_for_prefix("compression_", "schedule_evictions_")

    tokenizer = AutoTokenizer.from_pretrained(model)
    sample, = load_dataset('THUDM/LongBench', 'narrativeqa', split='test', streaming=True).take(1)

    token_len = len(tokenizer.encode(sample['context']))
    ratio = token_len / max_model_len
    trunc_sample = sample['context'][:int(len(sample['context'])//ratio)]

    split_sample = trunc_sample.split(".")
    haystack = ".".join(split_sample[:len(split_sample) // 2] + [NEEDLE] + split_sample[len(split_sample) // 2:])

    # formatted = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    #              "You will be given a body of text and a question to answer about it. "
    #              "Answer the question as reliably as possible.<|eot_id|>\n"
    #              "<|start_header_id|>user<|end_header_id|>\n\n"
    #              f"{haystack}\n\n{QUESTION}<|eot_id|>\n"
    #              "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #              f"{ANSWER_PREFIX}")

    formatted = f"{haystack}\n\n{QUESTION}\n\nThe best thing to do in San Francisco is"

    input_token_ids = tokenizer.encode(formatted)

    max_out_tokens = 20
    n_samples = 2
    vllm_outputs = vllm_model.generate_greedy_logprobs(
        max_out_tokens,
        0,
        prompt_token_ids=[deepcopy(input_token_ids) for _ in range(n_samples)],
        max_cache_tokens=256,
        protected_window_size=32,
        metric_collection_buffer_size=0,
        compress_once=True,
        compress_chunks=True,
        observation_context_len=128,
    )

    for out in vllm_outputs:
        response = out[1]
        print(response)

        assert CORRECT_ANSWER in response


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("chunk_size", [1008])  # -1 1008
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_chunked_compression(
    vllm_runner,
    checkpointer,
    model: str,
    dtype: str,
    chunk_size: int,
) -> None:

    # Don't checkpoint during profiling
    max_model_len = 10000
    checkpointer.set_condition(checkpoint_layer=lambda _: False)
    checkpointer.do_save = False
    checkpointer.do_validate = False
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        enable_chunked_prefill=True,
        block_size=16,
        even_layer_evict=False,
        control_layers=[],
        kv_head_bias_path=None,
        kv_head_bias_weight=50000,
        new_token_limit=100,
        compression_interval=1000,
        gpu_memory_utilization=0.9,
        # max_num_batched_tokens=chunk_size if chunk_size > 0 else max_model_len + 500,
        # max_num_seqs=min(256, chunk_size) if chunk_size > 0 else 256,
        max_num_batched_tokens=max_model_len,
        max_chunk_len=chunk_size,
        max_model_len=max_model_len + 500,
    )
    checkpointer.do_save = False
    checkpointer.do_validate = True
    checkpointer.override_output = True
    checkpointer.base_dir = './checkpoints'
    checkpointer.tolerance = 1e-2
    checkpointer.set_condition(checkpoint_layer=lambda x: x in CHECK_LAYERS)
    checkpointer.enable_for_prefix("compression_", "schedule_evictions_")

    tokenizer = AutoTokenizer.from_pretrained(model)
    sample, = load_dataset('THUDM/LongBench', 'narrativeqa', split='test', streaming=True).take(1)

    token_len = len(tokenizer.encode(sample['context']))
    ratio = token_len / max_model_len
    trunc_sample = sample['context'][:int(len(sample['context'])//ratio)]\
    # trunc_sample = sample['context'][:4500]

    split_sample = trunc_sample.split(".")
    haystack = ".".join(split_sample[:len(split_sample) // 2] + [NEEDLE] + split_sample[len(split_sample) // 2:])

    # formatted = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    #              "You will be given a body of text and a question to answer about it. "
    #              "Answer the question as reliably as possible.<|eot_id|>\n"
    #              "<|start_header_id|>user<|end_header_id|>\n\n"
    #              f"{haystack}\n\n{QUESTION}<|eot_id|>\n"
    #              "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #              f"{ANSWER_PREFIX}")

    formatted = f"{haystack}\n\n{QUESTION}\n\nThe best thing to do in San Francisco is"

    input_token_ids = tokenizer.encode(formatted)

    max_out_tokens = 20
    n_samples = 3
    vllm_outputs = vllm_model.generate_greedy_logprobs(
        max_out_tokens,
        0,
        prompt_token_ids=[deepcopy(input_token_ids) for _ in range(n_samples)],
        max_cache_tokens=256,
        protected_window_size=32,
        metric_collection_buffer_size=0,
        compress_once=True,
        compress_chunks=True,
        observation_context_len=128,
    )

    for out in vllm_outputs:
        response = out[1]
        print(response)

        assert CORRECT_ANSWER in response


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("chunk_size", [4032])  # -1 1008, 512
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_varied_seq_len(
    vllm_runner,
    checkpointer,
    model: str,
    dtype: str,
    chunk_size: int,
) -> None:

    # Don't checkpoint during profiling
    max_model_len = 10000
    checkpointer.set_condition(checkpoint_layer=lambda _: False)
    checkpointer.do_save = False
    checkpointer.do_validate = False
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        enable_chunked_prefill=True,
        block_size=16,
        even_layer_evict=False,
        control_layers=[],
        kv_head_bias_path=None,
        kv_head_bias_weight=50000,
        new_token_limit=100,
        compression_interval=1000,
        gpu_memory_utilization=0.9,
        # max_num_batched_tokens=chunk_size if chunk_size > 0 else max_model_len + 500,
        # max_num_seqs=min(256, chunk_size) if chunk_size > 0 else 256,
        max_num_batched_tokens=max_model_len,
        max_chunk_len=chunk_size,
        max_model_len=max_model_len + 500,
    )
    checkpointer.do_save = False
    checkpointer.do_validate = True
    checkpointer.override_output = True
    checkpointer.base_dir = './checkpoints'
    checkpointer.tolerance = 1e-2
    checkpointer.set_condition(checkpoint_layer=lambda x: x in CHECK_LAYERS)
    checkpointer.enable_for_prefix("compression_", "schedule_evictions_")

    tokenizer = AutoTokenizer.from_pretrained(model)
    n_samples = 2
    max_out_tokens = 20
    samples = load_dataset('THUDM/LongBench', 'narrativeqa', split='test', streaming=True).take(n_samples)
    max_cache_tokens = [224, 256]
    obs_ctx_lens = [16, 128]
    max_prompt_len = [max_model_len // 2, max_model_len]

    input_ids = []
    sample_params = []
    for sample, max_cache_tok, max_len, obs_ctx_len in zip(
        samples, max_cache_tokens, max_prompt_len, obs_ctx_lens):
        token_len = len(tokenizer.encode(sample['context']))
        ratio = token_len / max_len
        trunc_sample = sample['context'][:int(len(sample['context'])//ratio)]\
        # trunc_sample = sample['context'][:4500]

        split_sample = trunc_sample.split(".")
        haystack = ".".join(split_sample[:len(split_sample) // 2] + [NEEDLE] + split_sample[len(split_sample) // 2:])

        # formatted = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        #              "You will be given a body of text and a question to answer about it. "
        #              "Answer the question as reliably as possible.<|eot_id|>\n"
        #              "<|start_header_id|>user<|end_header_id|>\n\n"
        #              f"{haystack}\n\n{QUESTION}<|eot_id|>\n"
        #              "<|start_header_id|>assistant<|end_header_id|>\n\n"
        #              f"{ANSWER_PREFIX}")

        formatted = f"{haystack}\n\n{QUESTION}\n\nThe best thing to do in San Francisco is"

        input_token_ids = tokenizer.encode(formatted)
        input_ids.append(input_token_ids)
        sample_params.append(
            SamplingParams(temperature=0.0,
                           max_tokens=max_out_tokens,
                           logprobs=0,
                           max_cache_tokens=max_cache_tok,
                           protected_window_size=32,
                           metric_collection_buffer_size=0,
                           compress_once=True,
                           compress_chunks=True,
                           observation_context_len=obs_ctx_len)
        )

    vllm_outputs = vllm_model.generate_greedy_logprobs(
        prompt_token_ids=input_ids,
        sampling_param_list=sample_params,
    )

    for out in vllm_outputs:
        response = out[1]
        print(response)

        assert CORRECT_ANSWER in response


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("chunk_size", [1008])  # -1 1008
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_single_compression(
    vllm_runner,
    checkpointer,
    model: str,
    dtype: str,
    chunk_size: int,
) -> None:

    # Don't checkpoint during profiling
    max_model_len = 10000
    checkpointer.set_condition(checkpoint_layer=lambda _: False)
    checkpointer.do_save = False
    checkpointer.do_validate = False
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        enable_chunked_prefill=True,
        block_size=16,
        even_layer_evict=False,
        control_layers=[],
        kv_head_bias_path=None,
        kv_head_bias_weight=50000,
        new_token_limit=100,
        compression_interval=1000,
        gpu_memory_utilization=0.9,
        # max_num_batched_tokens=chunk_size if chunk_size > 0 else max_model_len + 500,
        # max_num_seqs=min(256, chunk_size) if chunk_size > 0 else 256,
        max_num_batched_tokens=max_model_len,
        max_chunk_len=chunk_size,
        max_model_len=max_model_len + 500,
    )
    checkpointer.do_save = False
    checkpointer.do_validate = True
    checkpointer.override_output = True
    checkpointer.base_dir = './checkpoints'
    checkpointer.tolerance = 1e-2
    checkpointer.set_condition(checkpoint_layer=lambda x: x in CHECK_LAYERS)
    checkpointer.enable_for_prefix("compression_", "schedule_evictions_")

    tokenizer = AutoTokenizer.from_pretrained(model)
    sample, = load_dataset('THUDM/LongBench', 'narrativeqa', split='test', streaming=True).take(1)

    # token_len = len(tokenizer.encode(sample['context']))
    # ratio = token_len / max_model_len
    # trunc_sample = sample['context'][:int(len(sample['context'])//ratio)]\
    trunc_sample = sample['context'][:4500]

    split_sample = trunc_sample.split(".")
    haystack = ".".join(split_sample[:len(split_sample) // 2] + [NEEDLE] + split_sample[len(split_sample) // 2:])

    # formatted = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    #              "You will be given a body of text and a question to answer about it. "
    #              "Answer the question as reliably as possible.<|eot_id|>\n"
    #              "<|start_header_id|>user<|end_header_id|>\n\n"
    #              f"{haystack}\n\n{QUESTION}<|eot_id|>\n"
    #              "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #              f"{ANSWER_PREFIX}")

    formatted = f"{haystack}\n\n{QUESTION}\n\nThe best thing to do in San Francisco is"

    input_token_ids = tokenizer.encode(formatted)

    max_out_tokens = 20
    n_samples = 3
    vllm_outputs = vllm_model.generate_greedy_logprobs(
        max_out_tokens,
        0,
        prompt_token_ids=[deepcopy(input_token_ids) for _ in range(n_samples)],
        max_cache_tokens=256, # 8992,
        protected_window_size=32,
        metric_collection_buffer_size=0,
        compress_once=True,
        compress_chunks=False,
        observation_context_len=128,
    )

    for out in vllm_outputs:
        response = out[1]
        print(response)

        assert CORRECT_ANSWER in response


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("chunk_size", [880])  # -1 1008
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
def test_no_compression(
    vllm_runner,
    checkpointer,
    model: str,
    dtype: str,
    chunk_size: int,
) -> None:

    # Don't checkpoint during profiling
    max_model_len = 10000
    checkpointer.set_condition(checkpoint_layer=lambda _: False)
    checkpointer.do_save = False
    checkpointer.do_validate = False
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        enable_kvcompress=True,
        enable_chunked_prefill=True,
        block_size=16,
        even_layer_evict=False,
        control_layers=[],
        kv_head_bias_path=None,
        kv_head_bias_weight=50000,
        new_token_limit=100,
        compression_interval=1000,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=chunk_size if chunk_size > 0 else max_model_len + 500,
        max_num_seqs=min(256, chunk_size) if chunk_size > 0 else 256,
        max_model_len=max_model_len + 500,
    )
    checkpointer.do_save = True
    checkpointer.do_validate = False
    checkpointer.base_dir = './checkpoints'
    checkpointer.set_condition(checkpoint_layer=lambda x: x in CHECK_LAYERS)
    checkpointer.enable_for_prefix("flash_prefix_")

    tokenizer = AutoTokenizer.from_pretrained(model)
    sample, = load_dataset('THUDM/LongBench', 'narrativeqa', split='test', streaming=True).take(1)

    token_len = len(tokenizer.encode(sample['context']))
    ratio = token_len / max_model_len
    trunc_sample = sample['context'][:int(len(sample['context'])//ratio)]

    split_sample = trunc_sample.split(".")
    haystack = ".".join(split_sample[:len(split_sample) // 2] + [NEEDLE] + split_sample[len(split_sample) // 2:])

    # formatted = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    #              "You will be given a body of text and a question to answer about it. "
    #              "Answer the question as reliably as possible.<|eot_id|>\n"
    #              "<|start_header_id|>user<|end_header_id|>\n\n"
    #              f"{haystack}\n\n{QUESTION}<|eot_id|>\n"
    #              "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #              f"{ANSWER_PREFIX}")

    formatted = f"{haystack}\n\n{QUESTION}\n\nThe best thing to do in San Francisco is"

    input_token_ids = tokenizer.encode(formatted)

    max_out_tokens = 20
    vllm_outputs = vllm_model.generate_greedy_logprobs(
        max_out_tokens,
        0,
        prompt_token_ids=[input_token_ids],
        max_cache_tokens=-1,
        protected_window_size=32,
        metric_collection_buffer_size=0,
        compress_once=True,
        observation_context_len=0,
    )

    response = vllm_outputs[0][1]

    assert CORRECT_ANSWER in response