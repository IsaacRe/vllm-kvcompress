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
ALL_CATEGORIES = ["Single-Doc. QA", "Multi-Doc. QA", "Summarization",
                  "Few-shot Learning", "Synthetic", "Code"]

parser = ArgumentParser()
parser.add_argument("--file", type=str, default="result.json")
parser.add_argument("--full-file", type=str, default=None)
parser.add_argument("--code-file", type=str, default=None)
parser.add_argument("--exp-id", type=str, default="w32_L2_cc")
parser.add_argument("--full-exp-id", type=str, default="w32_L2_cc")
parser.add_argument("--save-dir", type=str, default="./")
parser.add_argument("--ylim", type=str, nargs=2, default=("None", "None"))
args = parser.parse_args()

set_ymin, set_ymax = map(lambda x: None if x.lower() == "none" else float(x), args.ylim)

if args.full_file is None:
    args.full_file = args.file

if args.full_exp_id is None:
    args.full_exp_id = args.exp_id

print(args.full_exp_id)

results = json.load(open(args.file, 'r'))
full_results = json.load(open(args.full_file, 'r'))
code_results = {}
if args.code_file is not None:
    code_results = json.load(open(args.code_file, 'r'))

fig, ax = plt.subplots(2, 3, figsize=(12, 7))


for i, category in enumerate(ALL_CATEGORIES):
    x, y = i // 3, i % 3
    ax_ = ax[x, y]
    for dset, dset_cat in SUBTASK_CATEGORIES.items():
        if dset_cat == category:
            try:
                if category == "Code" and "70b" in args.file:
                    # full_score = full_results[f'{dset}-full_{args.full_exp_id}']
                    # dset_results = ([full_score] +
                    #         [results[f'{dset}-{cr}_{args.exp_id}']
                    #             for cr in COMPRESSION_RATE_STRINGS])
                    # ax_.plot(COMPRESSION_RATES, dset_results, label=dset)
                    full_score = code_results[f'{dset}-full_{args.full_exp_id}']
                    dset_results = ([100.] +
                            [code_results[f'{dset}-{cr}_{args.exp_id}'] / full_score * 100.
                                for cr in COMPRESSION_RATE_STRINGS])
                    ax_.plot(COMPRESSION_RATES, dset_results, label=dset)
                else:
                    full_score = full_results[f'{dset}-full_{args.full_exp_id}']
                    dset_results = ([100.] +
                            [results[f'{dset}-{cr}_{args.exp_id}'] / full_score * 100.
                                for cr in COMPRESSION_RATE_STRINGS])
                    ax_.plot(COMPRESSION_RATES, dset_results, label=dset)
            except:
                pass

    ymin, ymax = ax_.get_ylim()
    ymax = min(ymax, 107.0)
    if set_ymin is not None:
        ymin = set_ymin
    if set_ymax is not None:
        ymax = set_ymax
    # if category == "Code" and "70b" in args.file:
    #     ax_.set_ylabel("absolute performance")
    if False:
        pass
    else:
        ax_.set_ylim(ymin, ymax)
        if y == 0:
            ax_.set_ylabel("% performance")
    ax_.legend(loc='lower left')
    ax_.set_title(category)
    ax_.set_xscale('log')
    ax_.set_xticks(COMPRESSION_RATES)
    ax_.set_xticklabels(COMPRESSION_RATES)
    if x == 1:
        ax_.set_xlabel("compression rate")
    ax_.grid()


plt.savefig(f'{args.save_dir}/longbench_score_by_cr_all.jpg')
plt.savefig(f'{args.save_dir}/longbench_score_by_cr_all.pdf')
plt.show()
