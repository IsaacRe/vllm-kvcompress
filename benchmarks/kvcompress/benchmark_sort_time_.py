"""
torch.sort for single-dimensional tensors has unstable latency for large array lengths
(see out.png) during the first call, with certain specific lengths demonstrating consistently
higher latency than all other lengths. In subsequent calls (see out_warm.png) the latency is
stable and remains low until array lengths of around 192,000,000 (device dependent),
at which point it begins scaling linearly (probably due to SM's being maxed out). The discrepancy
doesn't seem to be due to device warm up since an initial call to torch.sort followed by
a cool-down period yields the same improvement in latency stability during subsequent calls.
"""
from argparse import ArgumentParser
from datetime import datetime
import torch
import math
import time
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--sort-dim-max", default=253_440_000, type=int)
parser.add_argument("--sort-dim-step", default=2_560_000, type=int)
parser.add_argument("--pad-to", default=253_440_000, type=int)
parser.add_argument("--dtype", choices=["int", "float"], default="int")
parser.add_argument("--device", default="cuda:0", type=str)


def plot_by_len(args):
    times = []
    total = args.sort_dim_max
    if args.dtype == "int":
        x = torch.tensor(
            np.random.choice(total, total, replace=False), dtype=torch.int).to(args.device)
    else:
        x = torch.rand(total).to(args.device)
    if args.pad_to:
        padded = torch.zeros((args.pad_to,)).type(x.dtype).to(x.device)
    slow_down = False
    for i in tqdm(range(args.sort_dim_step, args.sort_dim_max + 1, args.sort_dim_step)):
        if args.pad_to:
            padded[:i] = x[:i]
        start = datetime.now()
        if args.pad_to:
            sorted = padded.sort(dim=0).values[:i]
        else:
            sorted = x[:i].sort(dim=0).values
        times.append((i, (datetime.now() - start).total_seconds()))
        if times[-1][1] > 0.001 and not slow_down:
            print(f'first slow down at {times[-1][0]}')
            slow_down = True
        # print(sorted[:4], sorted[-4:])
    return times


def main():
    args = parser.parse_args()

    plot_by_len(args)
    # cooldown
    print('cooling down...')
    for i in tqdm(range(10)):
        time.sleep(1)
    times1 = plot_by_len(args)

    plt.plot(*zip(*times1), label="pad")

    # args.pad_to = None
    # times2 = plot_by_len(args)

    # plt.plot(*zip(*times2), label="no pad")

    # plt.legend()
    plt.grid()
    # plt.ylim(-0.0001, max(t for _, t in times1) + 0.0001)

    plt.title("torch.sort GPU runtime by array len")
    plt.xlabel("array len")
    plt.ylabel("runtime (s)")
    plt.savefig("out.png")


if __name__ == "__main__":
    main()
