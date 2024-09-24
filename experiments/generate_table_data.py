import json
import os.path
from argparse import ArgumentParser
import requests
API_URL = "https://datasets-server.huggingface.co/size?dataset=THUDM/LongBench"
def query_longbench_subtask_sizes():
    response = requests.get(API_URL)
    return {cfg["config"]: cfg["num_rows"] for cfg in
            response.json()["size"]["configs"]}
subset_sizes = query_longbench_subtask_sizes()

def is_complete(model, dset, cache_size, exp_id, backup=None):
    path = f'results/{model}/{dset}-{cache_size}_{exp_id}.jsonl'
    backup_path = f'results/{backup}/{dset}-{cache_size}_{exp_id}.jsonl'
    found = False
    if os.path.exists(path):
        nlines = len(open(path).readlines())
        if nlines == subset_sizes[dset]:
            found = True
    if not found:
        if os.path.exists(backup_path):
            nlines = len(open(backup_path).readlines())
            if nlines == subset_sizes[dset]:
                found = True
    return found


parser = ArgumentParser()
parser.add_argument('--model', type=str, default='mistral')
parser.add_argument('--backup-model', type=str, default=None)
parser.add_argument('--method', type=str, default='w32_L2')
parser.add_argument('--full', action='store_true')
args = parser.parse_args()

DATASET_NAMES = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

BLOCK_SIZE = 16
CACHE_SIZES = [128, 256, 512, 1024]
MIN_CACHE_SIZES = [i - BLOCK_SIZE for i in CACHE_SIZES]

with open(f'./results/{args.model}/result.json', 'r') as f:
    results = json.load(f)

with open(f'./results/{args.backup_model}/result.json', 'r') as f:
    backup_results = json.load(f)

for k, v in backup_results.items():
    results[k] = results.get(k, backup_results[k])

print(results)
for dset in DATASET_NAMES:
    print(dset[:6], end='\t')
print("avg.")

if args.full:
    tot = 0
    result_count = 0
    for dset in DATASET_NAMES:
        key = f'{dset}-full_{args.method}'
        if key not in results:
            print('null ', end='\t')
            continue
        print(results[key], end='\t')
        tot += results[key]
        result_count += 1
    if result_count == len(DATASET_NAMES):
        print("%.2f" % (tot / len(DATASET_NAMES)))
    else:
        print("null")

else:
    result_by_cache_size = {}
    print('max-cache:')
    for cs in CACHE_SIZES:
        tot = 0
        result_by_cache_size[cs] = []
        for dset in DATASET_NAMES:
            key = f'{dset}-{cs}_{args.method}'
            if not is_complete(args.model, dset, cs, args.method,
                               backup=args.backup_model) or key not in results:
                print('null ', end='\t')
                result_by_cache_size[cs].append(None)
                continue
            print(results[key], end='\t')
            tot += results[key]
            result_by_cache_size[cs].append(results[key])
        if not any(map(lambda x: x is None, result_by_cache_size[cs])):
            print("%.2f" % (tot / len(DATASET_NAMES)))
        else:
            print("null")
    print('min-cache:')
    for cs in MIN_CACHE_SIZES:
        tot = 0
        result_by_cache_size[cs] = []
        for dset in DATASET_NAMES:
            key = f'{dset}-{cs}_{args.method}'
            if not is_complete(args.model, dset, cs, args.method):
                print('null ', end='\t')
                result_by_cache_size[cs].append(None)
                continue
            print(results[key], end='\t')
            tot += results[key]
            result_by_cache_size[cs].append(results[key])
        if not any(map(lambda x: x is None, result_by_cache_size[cs])):
            print("%.2f" % (tot / len(DATASET_NAMES)))
        else:
            print("null")
    print('average:')
    for cs in CACHE_SIZES:
        tot = 0
        result_count = 0
        for i in range(len(result_by_cache_size[cs])):
            if (result_by_cache_size[cs][i] is None or
                result_by_cache_size[cs - BLOCK_SIZE][i] is None):
                print('null ', end='\t')
                continue
            avg = (result_by_cache_size[cs][i] +
                   result_by_cache_size[cs - BLOCK_SIZE][i]) / 2
            tot += avg
            result_count += 1
            print("%.2f" % avg, end='\t')
        if result_count == len(DATASET_NAMES):
            print("%.2f" % (tot / len(DATASET_NAMES)))
        else:
            print("null")
