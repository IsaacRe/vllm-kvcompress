outdir=benchmarks/results
mkdir -p $outdir
for input_len in 1000 2000 3000 4000 5000 6000 7000; do
    for output_len in 100 200 300 400 500; do
        for max_cache_tokens in 128 256 512 1024; do
            python3 benchmarks/benchmark_throughput.py --num-prompts 100 --input-len $input_len --output-len $output_len --enforce-eager --max-cache-tokens $max_cache_tokens --new-token-limit $(($input_len - 1)) --kvc-interval 1000000 --model NousResearch/Hermes-2-Theta-Llama-3-8B --only-prefill-metrics --max-model-len 8144 --enable-kvc > $outdir/llama3_${input_len}_${output_len}_${max_cache_tokens}.out
        done
    done
done
for input_len in 1000 2000 3000 4000 5000 6000 7000; do
    for output_len in 100 200 300 400 500; do
        python3 benchmarks/benchmark_throughput.py --num-prompts 100 --input-len $input_len --output-len $output_len --enforce-eager --max-cache-tokens $max_cache_tokens --new-token-limit $(($input_len - 1)) --kvc-interval 1000000 --model NousResearch/Hermes-2-Theta-Llama-3-8B --only-prefill-metrics --max-model-len 8144 > $outdir/llama3_${input_len}_${output_len}_full.out
    done
done