rm longbench_1_out.txt
python check_completion.py --exp-id w32_L2 --cache-sizes full --model llama3 > incomplete-full.txt
python check_completion.py --exp-id w8_L2_cc --cache-sizes varied --model llama3 > incomplete-varied.txt
python check_completion.py --exp-id w8_L2 --cache-sizes fixed --model llama3 > incomplete-fixed.txt
metric_aggregation=L2-sum
for dataset in $(cat datasets.txt);
do
	# by compression-rate
	for compression_rate in 1 2 4 8 16 32 64;
	do
		exp_id="${dataset}-${compression_rate}.0x_w8_L2_cc"
		if [ "$(cat incomplete-varied.txt | grep $exp_id)" ]; then
			echo "python3 run_longbench.py --protected-window-size 8 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 8 --dataset $dataset --compression-rate $compression_rate --metric-aggregation $metric_aggregation --continual-compression --model llama3" &>> longbench_1_out.txt
			python3 run_longbench.py --protected-window-size 8 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 8 --dataset $dataset --compression-rate $compression_rate --metric-aggregation $metric_aggregation --continual-compression --model llama3 &>> longbench_1_out.txt
		else
			echo "Skipping $exp_id..." >> longbench_1_out.txt
		fi
	done
	# full-cache
	exp_id="${dataset}-full_w32_L2"
	if [ "$(cat incomplete-full.txt | grep $exp_id)" ]; then
		echo "python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --dataset $dataset --max-cache-tokens -1 --metric-aggregation $metric_aggregation --model llama3" &>> longbench_1_out.txt
		python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --dataset $dataset --max-cache-tokens -1 --metric-aggregation $metric_aggregation --model llama3 &>> longbench_1_out.txt
	else
		echo "Skipping $exp_id..." >> longbench_1_out.txt
	fi
	# fixed-cache
	for max_cache_tokens in 1024 512 256 128;
	do
		exp_id="${dataset}-${max_cache_tokens}_w8_L2"
		if [ "$(cat incomplete-fixed.txt | grep $exp_id)" ]; then
			echo "python3 run_longbench.py --protected-window-size 8 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 8 --dataset $dataset --max-cache-tokens $max_cache_tokens --metric-aggregation $metric_aggregation --model llama3" &>> longbench_1_out.txt
			python3 run_longbench.py --protected-window-size 8 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 8 --dataset $dataset --max-cache-tokens $max_cache_tokens --metric-aggregation $metric_aggregation --model llama3 &>> longbench_1_out.txt
		else
			echo "Skipping $exp_id..." >> longbench_1_out.txt
		fi
	done
done