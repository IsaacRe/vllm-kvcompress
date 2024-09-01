import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='mistral')
parser.add_argument('--method', type=str, default='w32_L2')
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

CACHE_SIZES = [128, 256, 512, 1024]

with open(f'./results/{args.model}/result.json', 'r') as f:
    results = json.load(f)

print(results)
for dset in DATASET_NAMES:
    print(dset[:6], end='\t')
print()
for cs in CACHE_SIZES:
    for dset in DATASET_NAMES:
        key = f'{dset}-{cs}_{args.method}'
        if key not in results:
            print('null ', end='\t')
            continue
        print(results[key], end='\t')
    print()
