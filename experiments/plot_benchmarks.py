import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import os
from matplotlib.lines import Line2D

cm = plt.get_cmap('gist_rainbow')
color_cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
NUM_COLORS = 12
new_colors = NUM_COLORS - len(color_cycle)
color_cycle.extend([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
COMPRESSION_RATES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

parser = ArgumentParser()
parser.add_argument("--file", type=str, default="out.csv")
parser.add_argument("--plot-input-lens", type=int, nargs="+", default=[-1])
parser.add_argument("--max-input-len", type=int, default=None)
parser.add_argument("--min-input-len", type=int, default=None)
parser.add_argument("--plot-cr-mult", type=float, nargs="+", default=[-1.0])
parser.add_argument("--save-dir", type=str, default=None)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--gpu", type=str, default=None)
args = parser.parse_args()

save_dir = args.save_dir
if save_dir is None:
    save_dir = args.file.split('.')[0]
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(args.file)
df = df[~df.req_per_s.isna()]

df = df[df.max_cache_tokens.apply(lambda x: x.split('-')[0][:-1] != '')]
df['compression_rate'] = df.max_cache_tokens.apply(lambda x: int(x.split('-')[0][:-1])
                                                   if x.split('-')[0] != 'full' else -1)
df['row_id'] = [str(i) + '_' + str(cr) for i, cr in zip(df.input_len, df.compression_rate)]

# average runs with same compression rate
df = df.groupby('row_id').agg({'input_len': 'max', 'compression_rate': 'max',
                               'max_batch_size': 'max', 'tok_per_s': 'mean'})
print(df)

df__ = df.groupby('compression_rate').agg({'input_len': 'max'})
max_shared_input_len = df__.input_len.min()
if args.max_input_len is not None:
    max_shared_input_len = args.max_input_len
df = df[df.input_len <= max_shared_input_len]
if args.min_input_len is not None:
    df = df[df.input_len >= args.min_input_len]

max_compression_rate = df.compression_rate.max()
input_lengths = df.input_len.unique()
compression_rates = df.compression_rate[df.compression_rate != -1].unique()
if args.plot_cr_mult == [-1.0]:
    args.plot_cr_mult = compression_rates

fig, ax = plt.subplots()

# throughput by max compression rate
assert len(input_lengths) <= len(color_cycle), f'{len(input_lengths)} > {len(color_cycle)}'
mults = []
mult_crs = []
mult_baselines = []
for cr in sorted(args.plot_cr_mult):
    df_ = df[df.compression_rate == cr]
    if args.plot_input_lens != [-1]:
        df_ = df_[df_.input_len.apply(lambda x: x in args.plot_input_lens)]
    max_row = df_[df_.tok_per_s == df_.tok_per_s.max()].iloc[0]
    baseline = df[(df.input_len == max_row.input_len) & (df.compression_rate == -1)]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        mult = max_row.tok_per_s / baseline.tok_per_s
        mults.append(mult)
        mult_crs.append(max_row)
        mult_baselines.append(baseline)
for c, input_len in zip(color_cycle, sorted(input_lengths)):
    if args.plot_input_lens != [-1] and input_len not in args.plot_input_lens:
        continue
    df_ = df[(df.input_len == input_len) & (df.compression_rate != -1)]
    df_ = df_.sort_values('compression_rate')
    plt.plot(df_.compression_rate, df_.tok_per_s, label=input_len, c=c, alpha=0.7)

# plot thrpt multiplier
for max_mult, max_mult_cr, max_mult_baseline in zip(mults, mult_crs, mult_baselines):
    ax.plot([max_mult_cr.compression_rate * 1.05] * 2,
            [max_mult_baseline.tok_per_s, max_mult_cr.tok_per_s],
            linewidth=1,
            c='black')
    ax.text(max_mult_cr.compression_rate,
            max_mult_baseline.tok_per_s + (max_mult_cr.tok_per_s - max_mult_baseline.tok_per_s) / 2,
            s='%.2fx' % max_mult,
            fontsize=10,
            horizontalalignment='right')

xmin, xmax = ax.get_xlim()

for c, input_len in zip(color_cycle, sorted(input_lengths)):
    if args.plot_input_lens != [-1] and input_len not in args.plot_input_lens:
        continue
    baseline = df[(df.input_len == input_len) & (df.compression_rate == -1)]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        ax.plot([xmin, xmax], [baseline.tok_per_s] * 2, c=c, linestyle='--', linewidth=1, alpha=0.6)

legend_handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=legend_handles + [Line2D([0], [0], color='black', ls='--', label='vanilla vLLM')])
ax.set_xscale('log')
ax.set_xticks(COMPRESSION_RATES)
ax.set_xticklabels(COMPRESSION_RATES)
ax.set_xlim(xmin, xmax)
ax.grid()
ax.set_title(f"{args.gpu} throughput for varied input length")
ax.set_xlabel("compression rate")
ax.set_ylabel("throughput (tok/sec)")
plt.savefig(f'{save_dir}/throughtput_by_cr.jpg')
plt.savefig(f'{save_dir}/throughtput_by_cr.pdf')
plt.show()
plt.clf()

# thoughput by input length
for compression_rate in sorted(compression_rates):
    df_ = df[df.compression_rate == compression_rate].sort_values('input_len')
    df_ = df_[df_.input_len <= max_shared_input_len]
    plt.plot(df_.input_len, df_.tok_per_s, label=f'{compression_rate}x', alpha=0.7)
df_ = df[df.compression_rate == -1].sort_values('input_len')
plt.plot(df_.input_len, df_.tok_per_s, label='vanilla vLLM', linestyle='--', c='black')
plt.legend()
plt.grid()
plt.title(f"{args.gpu} - {args.model}")
plt.xlabel('input length')
plt.ylabel('throughput (tok/sec)')
plt.savefig(f'{save_dir}/throughput_by_len.pdf')
plt.savefig(f'{save_dir}/throughput_by_len.jpg')
plt.show()
plt.clf()

fig, ax = plt.subplots()

# max decoding batch by compression rate
for c, input_len in zip(color_cycle[1:], sorted(input_lengths)):
    if args.plot_input_lens != [-1] and input_len not in args.plot_input_lens:
        continue
    df_ = df[(df.input_len == input_len) & (df.compression_rate != -1)]
    df_ = df_.sort_values('compression_rate')
    ax.plot(df_.compression_rate, df_.max_batch_size, label=input_len, alpha=0.7, c=c)
xmin, xmax = plt.xlim()
for c, input_len in zip(color_cycle[1:], sorted(input_lengths)):
    if input_len not in args.plot_input_lens:
        continue
    df_ = df[(df.input_len == input_len) & (df.compression_rate == -1)]
    if len(df_) > 0:
        df_ = df_.iloc[0]
        ax.plot([xmin, xmax], [df_.max_batch_size] * 2, c=c, linestyle='--',
                 linewidth=1, alpha=0.6)

legend_handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=legend_handles + [Line2D([0], [0], color='black', ls='--', label='vanilla vLLM')])

ax.set_xscale('log')
ax.set_xticks(COMPRESSION_RATES)
ax.set_xticklabels(COMPRESSION_RATES)
ax.set_xlim(xmin, xmax)
ax.set_title(f"{args.gpu} - {args.model}")
ax.set_xlabel('compression rate')
ax.set_ylabel('batch size')
ax.grid()
plt.savefig(f'{save_dir}/max_batch_by_cr.pdf')
plt.savefig(f'{save_dir}/max_batch_by_cr.jpg')
plt.show()
plt.clf()

# max decoding batch by input length
for compression_rate in sorted(compression_rates):
    df_ = df[df.compression_rate == compression_rate].sort_values('input_len')
    plt.plot(df_.input_len, df_.max_batch_size, label=f'{compression_rate}x', alpha=0.7)
df_ = df[df.compression_rate == -1].sort_values('input_len')
if len(df_) > 0:
    plt.plot(df_.input_len, df_.max_batch_size, c='black', linestyle='--', label='vanilla vLLM')
plt.legend()
plt.title('Max decoding batch by input length')
plt.xlabel('input length')
plt.ylabel('batch size')
plt.grid()
plt.savefig(f'{save_dir}/max_batch_by_len.pdf')
plt.savefig(f'{save_dir}/max_batch_by_len.jpg')
plt.show()
