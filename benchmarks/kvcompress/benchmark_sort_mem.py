from argparse import ArgumentParser
from datetime import datetime
import torch
import math

parser = ArgumentParser()
parser.add_argument("--n-seqs", default=1, type=int)
parser.add_argument("--seq-len", default=1000, type=int)
parser.add_argument("--n-layer", default=32, type=int)
parser.add_argument("--n-kv-head", default=8, type=int)
parser.add_argument("--device", default="cuda:0", type=str)


def torch_sort(x, chunks=1):
    print(x.shape)
    incr = math.ceil(x.shape[0] / chunks)
    for i in range(0, x.shape[0], incr):
        print(x[i:i+incr].shape)
        x[i:i+incr].sort(dim=-1)


def main():
    args = parser.parse_args()
    baseline_mem = torch.cuda.max_memory_allocated(
        torch.device(args.device))
    x = torch.rand(args.n_seqs, args.seq_len * args.n_layer * args.n_kv_head)
    x = x.to(args.device)
    init_mem = torch.cuda.max_memory_allocated(
        torch.device(args.device))
    x.sort(dim=1)
    final_mem = torch.cuda.max_memory_allocated(
        torch.device(args.device))
    tensor_mem = init_mem - baseline_mem
    incremental_mem = final_mem - init_mem
    print(f"baseline: {baseline_mem}, at load: {init_mem}, at sort: {final_mem}")
    print(f"tensor memory: {tensor_mem}")
    print(f"incremental sort memory: {incremental_mem}")
    print(f"incremental sort per tensor mem: {incremental_mem / tensor_mem}")

if __name__ == "__main__":
    main()
