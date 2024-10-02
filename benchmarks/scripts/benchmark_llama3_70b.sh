RUN_ID="${RUN_ID:-0}"
output_len=500
run_id=$RUN_ID
outdir=benchmarks/results/smart-compression
mkdir -p $outdir
##### Finish 12000
for input_len in 500 1000 2000 4000 6000 8000 10000 12000 16000; do
    for compression_rate in 1 2 4 8 16 32 64; do
        echo "python3 benchmarks/benchmark_throughput.py --num-prompts 256 --input-len $input_len --output-len $output_len --enforce-eager --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 --max-model-len 33000 --compression-rate $compression_rate --protected-window-size 32 --max-kv-per-compression 50000000 --gpu-memory-utilization 0.96 --enable-kvc > $outdir/llama3_${input_len}_${output_len}_${compression_rate}x-${run_id}.out"
        python3 benchmarks/benchmark_throughput.py --num-prompts 256 --input-len $input_len --output-len $output_len --enforce-eager --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 --max-model-len 33000 --compression-rate $compression_rate --protected-window-size 32 --max-kv-per-compression 50000000 --gpu-memory-utilization 0.96 --enable-kvc > $outdir/llama3_${input_len}_${output_len}_${compression_rate}x-${run_id}.out
    done
done