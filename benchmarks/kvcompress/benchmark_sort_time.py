from argparse import ArgumentParser
from datetime import datetime
import torch
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--plot", action="store_true")
parser.add_argument("--n-seqs", default=1, type=int)
parser.add_argument("--seq-len", default=1000, type=int)
parser.add_argument("--n-layer", default=32, type=int)
parser.add_argument("--n-kv-head", default=8, type=int)
parser.add_argument("--n-sample", default=2, type=int)
parser.add_argument("--api", choices=["torch", "python"], default="torch", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--chunks", default=1, type=int)
parser.add_argument("--data-gen", default="randint", choices=["randint", "uniform"], type=str)


def torch_sort(x, chunks=1):
    incr = math.ceil(x.shape[0] / chunks)
    iters = 0
    shapes = []
    for i in range(0, x.shape[0], incr):
        x[i:i+incr] = x[i:i+incr].sort(dim=-1).values
        iters += 1
        shapes.append(x[i:i+incr].shape)
    print(f"num_iters: {iters}, shape: {shapes}")
    return x


def plot_by_len(args):
    times = []
    multiplier = args.n_seqs * args.n_layer * args.n_kv_head
    total = multiplier * args.seq_len
    x = torch.tensor(np.random.choice(total, total, replace=False)).to(args.device)
    for i in tqdm(range(args.seq_len)):
        size = i * multiplier
        start = datetime.now()
        x[:size].sort(dim=0)
        times.append((i * multiplier, (datetime.now() - start).total_seconds()))
    return times


def plot_by_len_x(x, multiplier, up_to):
    start_ = up_to - multiplier * 10
    for i in tqdm(range(start_, up_to + 1, multiplier)):
        y = x[:, :i]
        print(y.shape)
        start = datetime.now()
        y.sort(dim=1)
        print((i, (datetime.now() - start).total_seconds()))


def time_sort(batch_dim, sort_dim):
    x = torch.tensor(np.random.choice(sort_dim, sort_dim, replace=False))
    x = x[None].repeat(batch_dim, 1)
    start = datetime.now()
    x.sort(dim=1)
    print((datetime.now() - start).total_seconds())


def main():
    args = parser.parse_args()

    if args.plot:
        times = plot_by_len(args)
        plt.plot(*zip(*times))
        plt.savefig("out.png")
        return

    seconds = 0
    for i in range(args.n_sample):
        total = args.seq_len * args.n_layer * args.n_kv_head
        if args.data_gen == "randint":
            x = np.random.choice(total, total, replace=False)
            x = x[None].repeat(args.n_seqs, 0)
            x = torch.tensor(x)
        else:
            x = torch.rand(args.n_seqs, args.seq_len * args.n_layer * args.n_kv_head)
        # plot_by_len_x(x, args.n_layer * args.n_kv_head, args.seq_len * args.n_layer * args.n_kv_head)
        # return
        print(x[0,:10])
        x = x.to(args.device)
        # print(f"Iter {i} - generated randoms")
        start = datetime.now()
        if args.api == "python":
            sorted(x)
        else:
            out = torch_sort(x, chunks=args.chunks)
        seconds += (datetime.now() - start).total_seconds()
        print(out[0,:10])
        if args.api == "torch" and args.data_gen == "randint":
            assert (out == torch.arange(total).type(out.dtype).to(out.device)[None]).all()
        # print(f"Iter {i} - finished sorting")
    print(f"{seconds / args.n_sample}s")

if __name__ == "__main__":
    main()
