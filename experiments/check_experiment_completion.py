import os.path
from argparse import ArgumentParser
import requests
API_URL = "https://datasets-server.huggingface.co/size?dataset=THUDM/LongBench"
def query_longbench_subtask_sizes():
    response = requests.get(API_URL)
    return {cfg["config"]: cfg["num_rows"] for cfg in
            response.json()["size"]["configs"]}
subset_sizes = query_longbench_subtask_sizes()

# FIXED_CACHE_SIZES = ['112', '128', '240', '256', '496', '512', '1008', '1024']
FIXED_CACHE_SIZES = ['128', '256', '512', '1024']
COMPRESSION_RATES = ['2.0x', '4.0x', '8.0x', '16.0x', '32.0x', '64.0x']

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='mistral')
parser.add_argument('--exp-id', type=str, default='w32_L2')
parser.add_argument('--cache-sizes', type=str, choices=['fixed', 'varied', 'full'],
                    default='fixed')
args = parser.parse_args()

datasets_file = ('datasets-with-train-splits.txt'
                 if args.exp_id.endswith('_b') or args.exp_id.endswith('_rb')
                 else 'datasets.txt')
datasets = map(lambda x: x.strip(), open(datasets_file).readlines())

if args.cache_sizes == 'fixed':
    cache_sizes = FIXED_CACHE_SIZES
elif args.cache_sizes == 'varied':
    cache_sizes = COMPRESSION_RATES
else:
    cache_sizes = ['full']

for dset in datasets:
    for cache_size in cache_sizes:
        path = f'results/{args.model}/{dset}-{cache_size}_{args.exp_id}.jsonl'
        if os.path.exists(path):
            nlines = len(open(path).readlines())
            if nlines != subset_sizes[dset]:
                print(f'{path}:\t{nlines}')
        else:
            print(f'{path}:\tmissing')

print(cache_sizes)