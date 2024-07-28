from argparse import ArgumentParser
from datasets import load_dataset
from vllm import LLM, SamplingParams
import json

from util import load_tokenizer, seed_everything, build_chat, post_process, MODELS

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='mistral')
parser.add_argument('--kv-head-bias-path', type=str, default='../kv_head_bias_mistral.npz')
parser.add_argument('--dataset', type=str, default='hotpotqa')
parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max-cache-tokens', type=int, default=4096)
parser.add_argument('--max-model-len', type=int, default=4096)


def main(args):
    seed_everything(42)
    model_name = MODELS[args.model]
    model = LLM(
        model_name,
        dtype="half",
        enforce_eager=True,
        enable_kvcompress=True,
        block_size=16,
        kv_head_bias_path=args.kv_head_bias_path,
        kv_head_bias_weight=50,
        trust_remote_code=True,
        enable_chunked_prefill=False,
        tensor_parallel_size=1,
        disable_log_stats=True,
        max_model_len=args.max_model_len,
    )
    tokenizer = load_tokenizer(args.model)
    dset = load_dataset('THUDM/LongBench',
                        args.dataset,
                        split=args.split,
                        streaming=True)
    
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    prompt_format = dataset2prompt[args.dataset]
    max_output_tokens = dataset2maxlen[args.dataset]
    max_length = min(model.llm_engine.scheduler_config.max_num_batched_tokens,
                     model.llm_engine.model_config.max_model_len)

    inputs = []
    prompts = []
    final_prompts = []
    for json_obj in dset.take(1):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in args.model:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
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
        assert len(inputs[-1]) < max_length - max_output_tokens, f"{len(inputs[-1])=}, {max_length=}"

    sampling_params = SamplingParams(
        # max_tokens=max_output_tokens,
        temperature=0.0,
        max_cache_tokens=args.max_cache_tokens,
    )
    outputs = model.generate(prompt_token_ids=inputs,
                             sampling_params=sampling_params)
    for p, fp, output in zip(prompts, final_prompts, outputs):
        response = post_process(output.outputs[0].text, args.model)
        print(p)
        print(fp)
        print(response)
        print('\n====================================\n')
        for out in output.outputs:
            print(out)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)