import matplotlib.pyplot as plt
import json
from argparse import ArgumentParser

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
COMPRESSION_RATES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
COMPRESSION_RATE_STRINGS = ['2.0x', '4.0x', '8.0x', '16.0x', '32.0x', '64.0x']
ONE_SUBTASK_PER_CATEGORY = ['narrativeqa', 'hotpotqa', 'gov_report', 'trec',
                            'passage_retrieval_en', 'lcc']
SUBTASK_CATEGORIES = {
    "narrativeqa": "Single-Doc. QA",
    "qasper": "Single-Doc. QA",
    "multifieldqa_en": "Single-Doc. QA",
    "hotpotqa": "Multi-Doc. QA",
    "2wikimqa": "Multi-Doc. QA",
    "musique": "Multi-Doc. QA",
    "gov_report": "Summarization",
    "qmsum": "Summarization",
    "multi_news": "Summarization",
    "trec": "Few-shot Learning",
    "triviaqa": "Few-shot Learning",
    "samsum": "Few-shot Learning",
    "passage_count": "Synthetic",
    "passage_retrieval_en": "Synthetic",
    "lcc": "Code",
    "repobench-p": "Code",
}
ALL_SUBTASKS = map(lambda x: x.strip(), open('datasets.txt', 'r').readlines())

parser = ArgumentParser()
parser.add_argument("--file", type=str, default="result.json")
parser.add_argument("--full-file", type=str, default=None)
parser.add_argument("--code-file", type=str, default=None)
parser.add_argument("--exp-id", type=str, default="w32_L2_cc")
parser.add_argument("--full-exp-id", type=str, default="w32_L2_cc")
parser.add_argument("--subsets", type=str, nargs="+",
                    default=ONE_SUBTASK_PER_CATEGORY)
parser
parser.add_argument("--by-category", action="store_true")
parser.add_argument("--exclude-category", type=str, nargs="*", default=[])
parser.add_argument("--save-dir", type=str, default="./")
args = parser.parse_args()



if args.full_file is None:
    args.full_file = args.file

if args.full_exp_id is None:
    args.full_exp_id = args.exp_id

print(args.full_exp_id)

if args.subsets == ['one-per-category']:
    args.subsets = ONE_SUBTASK_PER_CATEGORY
elif args.subsets == ['all']:
    args.subsets = ALL_SUBTASKS

results = json.load(open(args.file, 'r'))
full_results = json.load(open(args.full_file, 'r'))
code_results = {}
if args.code_file:
    code_results = json.load(open(args.code_file, 'r'))

if args.by_category:
    category_results = {}
    category_subtask_cnt = {}
    for dset in args.subsets:
        try:
            category = SUBTASK_CATEGORIES[dset]
            if category in args.exclude_category:
                continue
            if category == 'Code' and '70b' in args.file:
                full_score = code_results[f'{dset}-full_{args.full_exp_id}']
                dset_results = [code_results[f'{dset}-{cr}_{args.exp_id}'] / full_score * 100.
                                for cr in COMPRESSION_RATE_STRINGS]
            else:
                full_score = full_results[f'{dset}-full_{args.full_exp_id}']
                dset_results = [results[f'{dset}-{cr}_{args.exp_id}'] / full_score * 100.
                                for cr in COMPRESSION_RATE_STRINGS]
            category_results[category] = [cr + dr
                                        for cr, dr in
                                        zip(category_results.get(category, [0] * len(COMPRESSION_RATE_STRINGS)),
                                                                dset_results)]
            category_subtask_cnt[category] = category_subtask_cnt.get(category, 0) + 1
        except:
            pass
    for c, r in category_results.items():
        results = [100.0] + [r_ / category_subtask_cnt[c] for r_ in r]
        plt.plot(COMPRESSION_RATES, results, label=c)
else:
    for dset in args.subsets:
        try:
            full_score = full_results[f'{dset}-full_{args.full_exp_id}']
            dset_results = ([100.] +
                    [results[f'{dset}-{cr}_{args.exp_id}'] / full_score * 100.
                        for cr in COMPRESSION_RATE_STRINGS])
            plt.plot(COMPRESSION_RATES, dset_results, label=dset)
        except:
            pass

ymin, ymax = plt.ylim()
ymax = min(ymax, 107.0)
plt.ylim(ymin, ymax)

plt.legend()
plt.xscale('log')
plt.xticks(COMPRESSION_RATES, labels=COMPRESSION_RATES)
plt.title("LongBench subtask performance")
plt.xlabel("compression rate")
plt.ylabel("% performance")
plt.grid()
plt.savefig(f'{args.save_dir}/longbench_score_by_cr.jpg')
plt.savefig(f'{args.save_dir}/longbench_score_by_cr.pdf')
plt.show()
