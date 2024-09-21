# Adapted from https://github.com/FasterDecoding/SnapKV/blob/main/experiments/LongBench/eval.py
import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mistral")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--min-tok-len', type=int, default=None)
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, lengths, all_classes, min_len):
    total_score = 0.
    total_preds = 0
    skips = 0
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        if min_len is not None and length < min_len:
            skips += 1
            continue
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
        total_preds += 1
    if total_preds == 0:
        return 0.
    return round(100 * total_score / total_preds, 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"results/{args.model}/"
    else:
        path = f"results/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        compression_cfg = filename.split('-')[-1].split('_')[0]
        compression_rate = (float(compression_cfg[:-1])
                            if compression_cfg.endswith('x') else None)
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        experiment = filename.split('.jsonl')[0]
        *dataset, compression = experiment.split('-')
        dataset = '-'.join(dataset)
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    continue
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if not predictions:
            continue
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            # get context length after compression
            if compression_rate is not None:
                lengths = [int(l / compression_rate) for l in lengths]
            score = scorer(dataset, predictions, answers, lengths, all_classes, args.min_tok_len)
        scores[experiment] = score
    if args.e:
        out_path = f"results/{args.model}/result.json"
    elif args.min_tok_len is not None:
        out_path = f"results/{args.model}/result-min{args.min_tok_len}.json"
    else:
        out_path = f"results/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
