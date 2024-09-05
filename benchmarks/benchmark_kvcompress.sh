# outdir=benchmarks/results/final
# mkdir -p $outdir
# for input_len in 1000 2000 3000 4000 6000 8000 12000 16000 24000 32000; do
#     for output_len in 500; do
#         for max_cache_tokens in 128 256 512 1024 2048; do
#             python3 benchmarks/benchmark_throughput.py --num-prompts 100 --input-len $input_len --output-len $output_len --enforce-eager --max-cache-tokens $max_cache_tokens --new-token-limit $(($input_len - 1)) --kvc-interval 1000000 --model NousResearch/Hermes-2-Theta-Llama-3-8B --max-model-len 8144 --metric-collection-buffer-size 10 --protected-window-size 32 --enable-kvc > $outdir/llama3_${input_len}_${output_len}_${max_cache_tokens}.out
#         done
#     done
# done
# for input_len in 1000 2000 3000 4000 5000 6000 7000; do
#     for output_len in 100 200 300 400 500; do
#         python3 benchmarks/benchmark_throughput.py --num-prompts 100 --input-len $input_len --output-len $output_len --enforce-eager --max-cache-tokens $max_cache_tokens --new-token-limit $(($input_len - 1)) --kvc-interval 1000000 --model NousResearch/Hermes-2-Theta-Llama-3-8B --max-model-len 8144 > $outdir/llama3_${input_len}_${output_len}_full.out
#     done
# done

outdir=benchmarks/results/compression-rate
mkdir -p $outdir
for input_len in 4000 6000 8000 10000 12000 14000 16000 18000; do  # complete: 2000
    for output_len in 500; do
        for compression_rate in 1 2 4 8 16 32 64; do
            python3 benchmarks/benchmark_throughput.py --num-prompts 256 --input-len $input_len --output-len $output_len --enforce-eager --model meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 19000 --compression-rate $compression_rate --new-token-limit $(($input_len - 1)) --kvc-interval 1000000 --metric-collection-buffer-size 10 --protected-window-size 32 --enable-kvc > $outdir/llama3_${input_len}_${output_len}_${compression_rate}x.out
        done
    done
done