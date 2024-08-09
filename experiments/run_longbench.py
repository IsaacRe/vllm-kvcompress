from argparse import ArgumentParser
from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
from tqdm.auto import tqdm
import os

from util import load_tokenizer, seed_everything, build_chat, post_process, MODELS

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='mistral')
parser.add_argument('--kv-head-bias-path', type=str, default='../kv_head_bias_mistral.npz')
parser.add_argument('--kv-head-bias-weight', type=int, default=50)
parser.add_argument('--dataset', type=str, default='hotpotqa')
parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max-cache-tokens', type=int, default=-1)
parser.add_argument('--max-kv-per-compression', type=int, default=50_000_000)
parser.add_argument('--protected-window-size', type=int, default=50)
parser.add_argument('--metric-collection-buffer-size', type=int, default=10)
parser.add_argument('--prefill-metric-collection-window-size', type=int, default=32)
parser.add_argument('--max-model-len', type=int, default=None)
parser.add_argument('--metric-aggregation', choices=['L1-sum', 'L1-avg', 'L2-sum', 'L2-avg'],
                    default='L2-sum')
parser.add_argument('--no-maxpool-metrics', action='store_false', dest='maxpool_metrics')

def main(args):
    seed_everything(42)
    model_name = MODELS[args.model]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    prompt_format = dataset2prompt[args.dataset]
    max_output_tokens = dataset2maxlen[args.dataset]

    model = LLM(
        model_name,
        dtype="half",
        enforce_eager=True,
        enable_kvcompress=True,
        compression_interval=max_output_tokens,  # only compress once after prefill
        new_token_limit=max_output_tokens,  # this should trigger compression after prefill
        block_size=16,
        kv_head_bias_path=args.kv_head_bias_path,
        kv_head_bias_weight=args.kv_head_bias_weight,
        trust_remote_code=True,
        enable_chunked_prefill=False,
        tensor_parallel_size=1,
        disable_log_stats=True,
        max_model_len=args.max_model_len,
        prefill_metric_collection_window_size=args.prefill_metric_collection_window_size,
        max_kv_per_compression=args.max_kv_per_compression,
        metric_aggregation=args.metric_aggregation,
        maxpool_metrics=args.maxpool_metrics,
    )
    max_length = min(model.llm_engine.scheduler_config.max_num_batched_tokens,
                     model.llm_engine.model_config.max_model_len)

    tokenizer = load_tokenizer(args.model)
    dset = load_dataset('THUDM/LongBench',
                        args.dataset,
                        split=args.split,
                        streaming=True)

    inputs = []
    prompts = []
    final_prompts = []
    json_objs = []
    print("Loading data...")
    for json_obj in tqdm(dset):
        json_objs.append(json_obj)
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in args.model:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length - max_output_tokens:
            half = int((max_length - max_output_tokens - 1)/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        prompts.append(prompt)
        if args.dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, args.model)
        final_prompts.append(prompt)
        if "chatglm3" in args.model:
            input = prompt.to(args.device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(args.device)
        inputs.append(input.input_ids[0].cpu().numpy().tolist())
        assert len(inputs[-1]) <= max_length - max_output_tokens, f"{len(inputs[-1])=}, {max_length - max_output_tokens=}"
        assert len(inputs[-1]) > max_output_tokens, "compression won't be triggered after prefill"

    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.0,
        max_cache_tokens=args.max_cache_tokens,
        protected_window_size=args.protected_window_size,
        metric_collection_buffer_size=args.metric_collection_buffer_size,
    )
    experiment_id = f"{args.max_cache_tokens if args.max_cache_tokens > 0 else 'full'}_w{args.prefill_metric_collection_window_size}_{args.metric_aggregation.split('-')[0]}_local"
    out_path = f"results/{args.model}/{args.dataset}-{experiment_id}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w+", encoding="utf-8") as f:
        for input_ids, fp, json_obj in zip(tqdm(inputs), final_prompts, json_objs):
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