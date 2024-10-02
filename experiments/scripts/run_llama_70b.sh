rm longbench_out.txt
metric_aggregation='L2-sum'
data_dir=../../PyramidKV/data

# full-cache
python check_completion.py --exp-id w32_L2_cc --cache-sizes full --model llama3-70b > incomplete.txt
for dataset in $(cat datasets.txt);
do
	exp_id="${dataset}-full_w32_L2_cc"
	if [ "$(cat incomplete.txt | grep $exp_id)" ]; then
		echo "python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --dataset $dataset --max-cache-tokens -1 --metric-aggregation $metric_aggregation --continual-compression --model llama3-70b --max-model-len 33000 --gpu-mem-util 0.96 --max-kv-per-compression 25_000_000" &>> longbench_out.txt
		python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --dataset $dataset --max-cache-tokens -1 --metric-aggregation $metric_aggregation --continual-compression --model llama3-70b --max-model-len 33000 --gpu-mem-util 0.96 --max-kv-per-compression 25_000_000 &>> longbench_out.txt
	else
		echo "Skipping $exp_id..." >> longbench_out.txt
	fi
done

# w32, continual-compression, by compression-rate
python check_completion.py --exp-id w32_L2_cc --cache-sizes varied --model llama3-70b > incomplete.txt
for dataset in $(cat datasets.txt);
do
	for compression_rate in 2 4 8 16 32 64;
	do
		exp_id="${dataset}-${compression_rate}.0x_w32_L2_cc"
		if [ "$(cat incomplete.txt | grep $exp_id)" ]; then
			echo "python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --dataset $dataset --compression-rate $compression_rate --metric-aggregation $metric_aggregation --continual-compression --model llama3-70b --max-model-len 33000 --gpu-mem-util 0.96 --max-kv-per-compression 25_000_000" --data-dir $data_dir &>> longbench_out.txt
			python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --dataset $dataset --compression-rate $compression_rate --metric-aggregation $metric_aggregation --continual-compression --model llama3-70b --max-model-len 33000 --gpu-mem-util 0.96 --max-kv-per-compression 25_000_000 --data-dir $data_dir &>> longbench_out.txt
		else
			echo "Skipping $exp_id..." >> longbench_out.txt
		fi
	done
done
