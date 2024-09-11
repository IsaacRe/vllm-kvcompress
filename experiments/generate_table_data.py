import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='mistral')
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

print(results)
for dset in DATASET_NAMES:
    print(dset[:6], end='\t')
print("avg.")

if args.full:
    tot = 0
    for dset in DATASET_NAMES:
        key = f'{dset}-full_{args.method}'
        if key not in results:
            print('null ', end='\t')
            continue
        print(results[key], end='\t')
        tot += results[key]
    print("%.2f" % (tot / len(DATASET_NAMES)))

else:
    result_by_cache_size = {}
    print('max-cache:')
    for cs in CACHE_SIZES:
        tot = 0
        result_by_cache_size[cs] = []
        for dset in DATASET_NAMES:
            key = f'{dset}-{cs}_{args.method}'
            if key not in results:
                print('null ', end='\t')
                continue
            print(results[key], end='\t')
            tot += results[key]
            result_by_cache_size[cs].append(results[key])
        print("%.2f" % (tot / len(DATASET_NAMES)))
    print('min-cache:')
    for cs in MIN_CACHE_SIZES:
        tot = 0
        result_by_cache_size[cs] = []
        for dset in DATASET_NAMES:
            key = f'{dset}-{cs}_{args.method}'
            if key not in results:
                print('null ', end='\t')
                result_by_cache_size[cs].append(None)
                continue
            print(results[key], end='\t')
            tot += results[key]
            result_by_cache_size[cs].append(results[key])
        print("%.2f" % (tot / len(DATASET_NAMES)))
    print('average:')
    for cs in CACHE_SIZES:
        tot = 0
        for i in range(len(result_by_cache_size[cs])):
            if (result_by_cache_size[cs][i] is None or
                result_by_cache_size[cs - BLOCK_SIZE][i] is None):
                print('null ', end='\t')
                continue
            avg = (result_by_cache_size[cs][i] +
                   result_by_cache_size[cs - BLOCK_SIZE][i]) / 2
            tot += avg
            print("%.2f" % avg, end='\t')
        print("%.2f" % (tot / len(DATASET_NAMES)))
