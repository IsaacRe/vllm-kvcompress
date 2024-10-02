"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
import uvloop
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser, merge_async_iterators
from vllm.benchmark import BENCHMARKER


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    num_scheduler_steps: int = 1,
    use_v2_block_manager: bool = False,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
    disable_async_output_proc: bool = False,
    max_batch_size: Optional[int] = None,
    enable_kvc: bool = False,
    kvc_rate: float = 1.0,
    protected_window_size: int = 50,
    metric_collection_buffer_size: int = 0,
    kv_head_bias_path: str = "./kv_head_bias.npz",
    kvc_interval: int = 1,
    max_kv_per_compression: int = 5_000_000,
    new_token_limit: int = -1,
    max_cache_tokens: int = -1,
    record_decoding_metrics: bool = True,
    metric_aggregation: str = "L2-sum",
    compress_once: bool = False,
    # speculative decoding
    speculative_model: str = None,
    speculative_model_quantization: str = None,
    spec_decoding_acceptance_method: str = 'typical_acceptane_sampler',
    typical_acceptance_sampler_posterior_threshold: float = 0.09,
    typical_acceptance_sampler_posterior_alpha: float = 0.3,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
        num_scheduler_steps=num_scheduler_steps,
        use_v2_block_manager=use_v2_block_manager,
        disable_async_output_proc=disable_async_output_proc,
        max_num_seqs=max_batch_size,
        enable_kvcompress=enable_kvc,
        target_compression_rate=kvc_rate,
        protected_window_size=protected_window_size,
        metric_collection_buffer_size=metric_collection_buffer_size,
        kv_head_bias_path=kv_head_bias_path,
        compression_interval=kvc_interval,
        max_kv_per_compression=max_kv_per_compression,
        new_token_limit=new_token_limit,
        max_cache_tokens=max_cache_tokens,
        record_decoding_metrics=record_decoding_metrics,
        metric_aggregation=metric_aggregation,
        # speculative decoding
        speculative_model=speculative_model,
        speculative_model_quantization=speculative_model_quantization,
        spec_decoding_acceptance_method=spec_decoding_acceptance_method,
        typical_acceptance_sampler_posterior_threshold=typical_acceptance_sampler_posterior_threshold,
        typical_acceptance_sampler_posterior_alpha=typical_acceptance_sampler_posterior_alpha,
    )

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
                max_cache_tokens=max_cache_tokens,
                target_compression_rate=kvc_rate,
                protected_window_size=protected_window_size,
                compress_once=compress_once,
            ))

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    # for output in outputs:
    #     assert len(output.outputs[0].token_ids) == output_len, (
    #         f"{len(output.outputs[0].token_ids)=}, {output_len=}")
    return end - start, llm.llm_engine.scheduler[0].max_decoding_batch


async def run_vllm_async(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    num_scheduler_steps: int = 1,
    use_v2_block_manager: bool = False,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
    disable_async_output_proc: bool = False,
    max_batch_size: Optional[int] = None,
    enable_kvc: bool = False,
    kvc_rate: float = 1.0,
    protected_window_size: int = 50,
    metric_collection_buffer_size: int = 0,
    kv_head_bias_path: str = "./kv_head_bias.npz",
    kvc_interval: int = 1,
    max_kv_per_compression: int = 5_000_000,
    new_token_limit: int = -1,
    max_cache_tokens: int = -1,
    record_decoding_metrics: bool = True,
    metric_aggregation: str = "L2-sum",
    compress_once: bool = False,
) -> float:
    from vllm import SamplingParams
    engine_args = AsyncEngineArgs(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
        num_scheduler_steps=num_scheduler_steps,
        use_v2_block_manager=use_v2_block_manager,
        disable_async_output_proc=disable_async_output_proc,
        worker_use_ray=False,
        engine_use_ray=False,
        disable_log_requests=True,
    )

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:

        # Add the requests to the engine.
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []
        for prompt, _, output_len in requests:
            prompts.append(prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=0.0 if use_beam_search else 1.0,
                    top_p=1.0,
                    use_beam_search=use_beam_search,
                    ignore_eos=True,
                    max_tokens=output_len,
                ))

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()
        return end - start


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve
    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
    return end - start


REAL_TEXT_PROMPT = """John Jeremy Thorpe (29 April 1929 – 4 December 2014) was a British politician who served as the Member of Parliament for North Devon from 1959 to 1979, and as leader of the Liberal Party from 1967 to 1976. In May 1979, he was tried at the Old Bailey on charges of conspiracy and incitement to murder his ex-boyfriend Norman Scott, a former model. Thorpe was acquitted on all charges, but the case, and the furore surrounding it, ended his political career.

Thorpe was the son and grandson of Conservative MPs, but decided to align with the small and ailing Liberal Party. After reading Law at Oxford University he became one of the Liberals' brightest stars in the 1950s. He entered Parliament at the age of 30, rapidly made his mark, and was elected party leader in 1967. After an uncertain start during which the party lost ground, Thorpe capitalised on the growing unpopularity of the Conservative and Labour parties to lead the Liberals through a period of electoral success. This culminated in the general election of February 1974, when the party won 6 million votes. Under the first-past-the-post electoral system this gave them only 14 seats, but in a hung parliament, no party having an overall majority, Thorpe was in a strong position. He was offered a cabinet post by the Conservative prime minister, Edward Heath, if he would bring the Liberals into a coalition. His price for such a deal, reform of the electoral system, was rejected by Heath, who resigned in favour of a minority Labour government.

The February 1974 election was the high-water mark of Thorpe's career. Thereafter his and his party's fortunes declined, particularly from late 1975 when rumours of his involvement in a plot to murder Norman Scott began to multiply. Thorpe resigned the leadership in May 1976 when his position became untenable. When the matter came to court three years later, Thorpe chose not to give evidence to avoid being cross-examined by counsel for the prosecution. This left many questions unanswered; despite his acquittal, Thorpe was discredited and did not return to public life. From the mid-1980s he was disabled by Parkinson's disease. During his long retirement he gradually recovered the affections of his party, and by the time of his death was honoured by a later generation of leaders, who drew attention to his record as an internationalist, a supporter of human rights and an opponent of apartheid and all forms of racism."""
CHAR_PER_TOK_APPROX = 2


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.real_text:
        requests = [(REAL_TEXT_PROMPT[:args.input_len * CHAR_PER_TOK_APPROX],
                     args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    elif args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    if args.compression_rate is not None:
        block_size = 16
        args.max_cache_tokens = max(128, ((int(args.input_len / args.compression_rate) - 1) // block_size * block_size))

    if args.backend == "vllm":
        run_args = [
            requests, args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.trust_remote_code, args.dtype, args.max_model_len,
            args.enforce_eager, args.kv_cache_dtype,
            args.quantization_param_path, args.device,
            args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.distributed_executor_backend,
            args.gpu_memory_utilization, args.num_scheduler_steps,
            args.use_v2_block_manager, args.download_dir, args.load_format,
            args.disable_async_output_proc, args.max_batch_size, args.enable_kvc,
            args.kvc_rate, args.protected_window_size, args.metric_collection_buffer_size,
            args.kv_head_bias_path, args.kvc_interval, args.max_kv_per_compression,
            args.new_token_limit, args.max_cache_tokens,
            args.record_decoding_metrics, args.metric_aggregation,
            args.compress_once,
            # speculative decoding
            args.speculative_model, args.speculative_model_quantization,
            args.spec_decoding_acceptance_method,
            args.typical_acceptance_sampler_posterior_threshold,
            args.typical_acceptance_sampler_posterior_alpha,
        ]

        if args.async_engine:
            run_args.append(args.disable_frontend_multiprocessing)
            elapsed_time = uvloop.run(run_vllm_async(*run_args))
            max_decoding_batch = None
        else:
            elapsed_time, max_decoding_batch = run_vllm(*run_args)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if args.benchmark_input_only:
        total_num_tokens = sum(prompt_len
                               for _, prompt_len, _ in requests)
    elif args.benchmark_output_only:
        total_num_tokens = sum(output_len
                               for _, _, output_len in requests)
    else:
        total_num_tokens = sum(prompt_len + output_len
                               for _, prompt_len, output_len in requests)
    print(f"Max decoding batch: {max_decoding_batch}")
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    if args.latency_breakdown:
        BENCHMARKER.summarize()

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help='device type for vLLM execution, supporting CUDA, OpenVINO and '
        'CPU.')
    parser.add_argument(
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.")
    parser.add_argument("--use-v2-block-manager",
                        action='store_true',
                        help="Enable block manager v2.")
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="Enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--load-format',
        type=str,
        default=EngineArgs.load_format,
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    parser.add_argument(
        "--disable-async-output-proc",
        action='store_true',
        default=False,
        help="Disable async output processor for vLLM backend.")
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=256,
        help='set max batch size for vLLM backend',
    )
    parser.add_argument(
        "--benchmark-input-only",
        type=int,
        help='Only use input tokens when computing tok/sec'
    )
    parser.add_argument(
        "--benchmark-output-only",
        action="store_true",
        help='Only use output tokens when computing tok/sec'
    )
    parser.add_argument(
        "--kvc-rate",
        type=float,
        default=1.0,
        help="KV cache compression rate",
    )
    parser.add_argument(
        "--enable-kvc",
        action="store_true",
        help="Enable KV cache compression",
    )
    parser.add_argument(
        "--protected-window-size",
        type=int,
        default=50,
        help="Protected window size for KV cache compression",
    )
    parser.add_argument(
        '--metric-collection-buffer-size',
        type=int,
        default=0,
        help="Buffer length for collecting KV metric",
    )
    parser.add_argument(
        "--kv-head-bias-path",
        type=str,
        default=None,
        help="Path to KV head bias for KV cache compression",
    )
    parser.add_argument(
        "--kvc-interval",
        type=int,
        default=1_000_000,
        help="Compress KV cache every n iterations",
    )
    parser.add_argument(
        "--max-kv-per-compression",
        type=int,
        default=5_000_000,
        help="Max number of KVs per compression",
    )
    parser.add_argument(
        '--new-token-limit',
        type=int,
        default=-1,
        help='Max number of tokens that can be added before compression '
        'is forced',
    )
    parser.add_argument(
        '--max-cache-tokens',
        type=int,
        default=-1,
        help='Number of tokens to compress to',
    )
    parser.add_argument(
        '--only-prefill-metrics',
        action='store_false',
        dest='record_decoding_metrics',
        help='Disable KV metric collection during decoding',
    )
    parser.add_argument(
        '--metric-aggregation',
        choices=['L1-sum', 'L1-avg', 'L2-sum', 'L2-avg'],
        default='L2-sum',
        help='Aggregation used for KV metrics',
    )
    parser.add_argument(
        '--compress-once',
        action='store_true',
        help='Whether to compress each each sequence only '
        'once after prefill',
    )
    parser.add_argument(
        '--compression-rate',
        type=float,
        default=None,
        help='Configure max_cache_tokens as a fraction of input length',
    )
    parser.add_argument(
        '--speculative-model',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--speculative-model-quantization',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--spec-decoding-acceptance-method',
        type=str,
        choices=['typical_acceptance_sampler', 'rejection_sampler'],
        default='typical_acceptance_sampler'
    )
    parser.add_argument(
        '--typical-acceptance-sampler-posterior-threshold',
        type=float,
        default=0.09,
    )
    parser.add_argument(
        '--typical-acceptance-sampler-posterior-alpha',
        type=float,
        default=0.3,  # sqrt of posterior threshold
    )
    parser.add_argument(
        '--real-text',
        action='store_true',
    )
    parser.add_argument(
        '--latency-breakdown',
        action='store_true',
    )
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
