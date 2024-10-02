RUN_ID="${RUN_ID:-0}"
outdir=benchmarks/results/smart-compression
mkdir -p $outdir
output_len=500
run_id=$RUN_ID
for input_len in 500 1000 2000 4000 6000 8000 10000 12000 14000; do
    for compression_rate in 1 2 4 8 16 32 64; do
        echo "python3 benchmarks/benchmark_throughput.py --num-prompts 256 --input-len $input_len --output-len $output_len --enforce-eager --model meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 19000 --compression-rate $compression_rate --protected-window-size 32 --enable-kvc > $outdir/llama3_${input_len}_${output_len}_${compression_rate}x-${run_id}.out"
        python3 benchmarks/benchmark_throughput.py --num-prompts 256 --input-len $input_len --output-len $output_len --enforce-eager --model meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 19000 --compression-rate $compression_rate --protected-window-size 32 --enable-kvc > $outdir/llama3_${input_len}_${output_len}_${compression_rate}x-${run_id}.out
    done
done
