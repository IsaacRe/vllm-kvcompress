from flash_attn import flash_attn_varlen_func
import torch
from datetime import datetime
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--n-warmup", type=int, default=5)
parser.add_argument("--n-trials", type=int, default=2)


def test_flash_attn_varlen_output(args):
    NUM_WARMUP = args.n_warmup
    NUM_TRIALS = args.n_trials
    dtype = torch.float16
    d = 128
    seqlen_q, seqlen_k = args.seq_len, args.seq_len
    alibi = False
    mha_type = "gqa"
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    alibi_slopes = None
    batch_size = args.batch_size
    nheads = 32
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 8)
    assert nheads % nheads_k == 0
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(
        batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype
    )
    q_reshape = q.view(-1, nheads, d)
    k_reshape = k.view(-1, nheads_k, d)
    v_reshape = v.view(-1, nheads_k, d)

    print(f'hi: {q_reshape.shape}')


    q_start_loc = (torch.arange(0, seqlen_q * batch_size, seqlen_q)
                        .type(torch.long)
                        .to(device))
    k_start_loc = (torch.arange(0, seqlen_k * batch_size, seqlen_k)
                        .type(torch.long)
                        .to(device))

    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
    else:
        alibi_slopes = None

    total_t = 0
    for i in range(NUM_WARMUP + NUM_TRIALS):
        start_t = datetime.now()
        attn_weights_ref = (
            q.transpose(-3, -2) @
            k.repeat(1, 1, nheads // nheads_k, 1).transpose(-3, -2).transpose(-2, -1)
        ).softmax(dim=-1)
        out = attn_weights_ref @ v.repeat(1, 1, nheads // nheads_k, 1).transpose(-3, -2)
        t = (datetime.now() - start_t).total_seconds() * 1000
        if i > NUM_WARMUP:
            total_t += t

    print(f"naive time: {total_t / NUM_TRIALS}ms")

    total_t = 0
    for i in range(NUM_WARMUP + NUM_TRIALS):
        start_t = datetime.now()
        out = flash_attn_varlen_func(
            q=q_reshape,
            k=k_reshape,
            v=v_reshape,
            cu_seqlens_q=q_start_loc.type(torch.int),
            cu_seqlens_k=k_start_loc.type(torch.int),
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            softmax_scale=1.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
        )
        t = (datetime.now() - start_t).total_seconds() * 1000
        if i > NUM_WARMUP:
            total_t += t
    
    print(f"flash-attn time: {total_t / NUM_TRIALS}ms")


if __name__ == "__main__":
    args = parser.parse_args()
    test_flash_attn_varlen_output(args)
