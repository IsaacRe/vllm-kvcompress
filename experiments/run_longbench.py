from argparse import ArgumentParser
from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
from tqdm.auto import tqdm
import os

from util import load_tokenizer, seed_everything, build_chat, post_process, MODELS

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='mistral')
parser.add_argument('--kv-head-bias-path', type=str, default=None)
parser.add_argument('--kv-head-bias-weight', type=int, default=0)
parser.add_argument('--dataset', type=str, default='hotpotqa')
parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max-cache-tokens', type=int, default=-1)
parser.add_argument('--max-kv-per-compression', type=int, default=50_000_000)
parser.add_argument('--protected-window-size', type=int, default=50)
parser.add_argument('--metric-collection-buffer-size', type=int, default=0)
parser.add_argument('--prefill-metric-collection-window-size', type=int, default=32)
parser.add_argument('--prefill-metric-collection-block-size', type=int, default=4096)
parser.add_argument('--max-model-len', type=int, default=None)
parser.add_argument('--metric-aggregation', choices=['L1-sum', 'L1-avg', 'L2-sum', 'L2-avg'],
                    default='L2-sum')
parser.add_argument('--no-maxpool-metrics', action='store_false', dest='maxpool_metrics')
parser.add_argument('--continual-compression', action='store_false', dest='compress_once')
parser.add_argument('--compression-rate', default=None, type=float)
parser.add_argument('--relative-kv-head-bias', action='store_true')
parser.add_argument('--gpu-mem-util', type=float, default=0.7)
parser.add_argument('--n-rows', type=int, default=-1)
parser.add_argument('--min-cache-tokens', type=int, default=128)
parser.add_argument('--test-row', type=int, default=None)
parser.add_argument('--run-id', type=str, default=None)
parser.add_argument('--data-dir', type=str, default=None)

def main(args):
    if args.compression_rate is not None:
        assert args.max_cache_tokens < 0, "cannot specify both compresion_rate and max_cache_tokens"

    seed_everything(42)
    model_name = MODELS[args.model]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    prompt_format_query = args.dataset
    # Llama 70B without compression performs poorly on coding subsets with og template
    if args.model == 'llama3-70b' and args.dataset in ['lcc', 'repobench-p']:
        prompt_format_query = f'{args.dataset}-llama_3_1_70b'
    prompt_format = dataset2prompt[prompt_format_query]
    max_output_tokens = dataset2maxlen[args.dataset]

    if args.kv_head_bias_path is None:
        args.kv_head_bias_path = f'kv-bias/kv_head_bias_{args.model}-{args.dataset}.npz'

    if args.kv_head_bias_weight == 0:
        args.kv_head_bias_path = None

    block_size = 16

    model = LLM(
        model_name,
        dtype="half",
        enforce_eager=True,
        enable_kvcompress=True,
        block_size=block_size,
        kv_head_bias_path=args.kv_head_bias_path,
        kv_head_bias_weight=args.kv_head_bias_weight,
        trust_remote_code=True,
        enable_chunked_prefill=False,
        tensor_parallel_size=1,
        disable_log_stats=True,
        max_model_len=args.max_model_len,
        prefill_metric_collection_window_size=args.prefill_metric_collection_window_size,
        prefill_metric_collection_block_size=args.prefill_metric_collection_block_size,
        max_kv_per_compression=args.max_kv_per_compression,
        metric_aggregation=args.metric_aggregation,
        maxpool_metrics=args.maxpool_metrics,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    # SnapKV sets max input length per model in their experiments
    max_prompt_length = model2maxlen[args.model]
    max_model_prompt_length = min(model.llm_engine.scheduler_config.max_num_batched_tokens,
                                  model.llm_engine.model_config.max_model_len)
    max_prompt_length = min(max_prompt_length, max_model_prompt_length - dataset2maxlen[args.dataset])

    tokenizer = load_tokenizer(args.model)

    def _load_longbench(subset, dir):
        records = []
        with open(f"{dir}/LongBench/{subset}.jsonl") as fp:
            for line in fp:
                records.append(json.loads(line))
        return records

    if args.data_dir is None:
        dset = load_dataset('THUDM/LongBench',
                            args.dataset,
                            split=args.split,
                            streaming=True)
    else:
        dset = _load_longbench(args.dataset, args.data_dir)

    inputs = []
    prompts = []
    final_prompts = []
    json_objs = []
    print("Loading data...")
    if args.n_rows > 0:
        dset = dset.take(args.n_rows)
    elif args.test_row is not None:
        dset = [list(dset.take(args.test_row + 1))[-1]]
    max_len = 0
    for json_obj in tqdm(dset):
        json_objs.append(json_obj)
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in args.model:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_prompt_length:
            half = int((max_prompt_length)/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        prompts.append(prompt)
        if args.dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p", "passage_count"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, args.model)
        final_prompts.append(prompt)
        if "chatglm3" in args.model:
            input = prompt.to(args.device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(args.device)
        inputs.append(input.input_ids[0].cpu().numpy().tolist())
        assert len(inputs[-1]) <= max_model_prompt_length, f"{len(inputs[-1])=}, {max_prompt_length=}, {max_model_prompt_length=}"
        max_len = max(len(inputs[-1]), max_len)

    print(f"{max_len=}")

    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        min_tokens=1,
        temperature=0.0,
        max_cache_tokens=args.max_cache_tokens,
        protected_window_size=args.protected_window_size,
        metric_collection_buffer_size=args.metric_collection_buffer_size,
        compress_once=args.compress_once,
    )
    cache_size_id = str(args.max_cache_tokens) if args.max_cache_tokens > 0 else (f"{args.compression_rate}x" if args.compression_rate is not None else 'full')
    experiment_id = f"{cache_size_id}_w{args.prefill_metric_collection_window_size}_{args.metric_aggregation.split('-')[0]}{('_rb' if args.relative_kv_head_bias else '_b') if args.kv_head_bias_weight > 0 else ''}{'_cc' if not args.compress_once else ''}"
    if args.run_id is not None:
        experiment_id = f"{experiment_id}_{args.run_id}"
    out_path = f"results/{args.model}/{args.dataset}-{experiment_id}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w+", encoding="utf-8") as f:
        for input_ids, fp, json_obj in zip(tqdm(inputs), final_prompts, json_objs):
            model.llm_engine.scheduler[0].block_manager.reinit()
            if args.compression_rate is not None:
                # subtract block size from max cache tokens to avoid going below the eviction requirement
                # due to evicting partially filled blocks
                sampling_params.max_cache_tokens = max(args.min_cache_tokens, ((int(len(input_ids) / args.compression_rate) - 1) // block_size * block_size))
            if args.relative_kv_head_bias:
                # multiply bias by output length
                output_len = len(tokenizer.encode(json_obj["answers"][0], add_special_tokens=False))
                bias_multiplier = output_len / 512
                # TODO make bias weight a sampling param
                model.llm_engine.kvcompress_state.kv_metrics.kv_metric_bias_weight *= bias_multiplier
            print(f"Using max_cache_tokens={sampling_params.max_cache_tokens}, input_len={len(input_ids)}")
            output = model.generate(prompt_token_ids=[input_ids],
                                    sampling_params=sampling_params)
            response = post_process(output[0].outputs[0].text, args.model)
            print('...' + fp[-200:])
            print(f'({len(input_ids)} tokens)')
            print(response)
            print(json_obj["answers"])
            print('\n====================================\n')
            json.dump({"pred": response, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        # for out in output.outputs:
        #     print(out)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)