for metric_aggregation in 'L2-sum' 'L1-sum';
do
	for dataset in $(cat datasets.txt);
	do
		for max_cache_tokens in -1 1024 512 256 128;
		do
			python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --kv-head-bias-weight 0 --dataset $dataset --max-cache-tokens $max_cache_tokens --metric-aggregation $metric_aggregation
		done
	done
done