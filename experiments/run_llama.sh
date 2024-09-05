# w32, continual-compression, no bias, by compression-rate
for metric_aggregation in 'L2-sum';
do
	for dataset in $(cat datasets.txt);
	do
		for compression_rate in 2 4 8 16 32 64;
		do
			python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --kv-head-bias-weight 0 --dataset $dataset --compression-rate $compression_rate --metric-aggregation $metric_aggregation --continual-compression --model llama3
		done
		python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --kv-head-bias-weight 0 --dataset $dataset --max-cache-tokens -1 --metric-aggregation $metric_aggregation --continual-compression --model llama3
	done
done
# w32, continual-compression, no bias
for metric_aggregation in 'L2-sum';
do
	for dataset in $(cat datasets.txt);
	do
		for max_cache_tokens in 1024 1008 512 496 256 240 128 112;
		do
			python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --kv-head-bias-weight 0 --dataset $dataset --max-cache-tokens $max_cache_tokens --metric-aggregation $metric_aggregation --continual-compression --model llama3
		done
	done
done
# w32, with bias
# for metric_aggregation in 'L2-sum';
# do
# 	for dataset in $(cat datasets.txt | grep qmsum -A 50);
# 	do
# 		for max_cache_tokens in 1024 1008 512 496 256 240 128 112;
# 		do
# 			python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --kv-head-bias-weight 1 --dataset $dataset --max-cache-tokens $max_cache_tokens --metric-aggregation $metric_aggregation
# 		done
# 	done
# done
# w32, no bias
for metric_aggregation in 'L2-sum';
do
	for dataset in $(cat datasets.txt);
	do
		for max_cache_tokens in 1024 1008 512 496 256 240 128 112;
		do
			python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 0 --prefill-metric-collection-window-size 32 --kv-head-bias-weight 0 --dataset $dataset --max-cache-tokens $max_cache_tokens --metric-aggregation $metric_aggregation --model llama3
		done
	done
done
# full query range, no bias
# for metric_aggregation in 'L2-sum';
# do
# 	# for dataset in $(cat datasets.txt);
# 	# do
# 	# 	for max_cache_tokens in 1024 512 256 128;
# 	# 	do
# 	# 		python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 10 --prefill-metric-collection-window-size 33000 --kv-head-bias-weight 0 --dataset $dataset --max-cache-tokens $max_cache_tokens --metric-aggregation $metric_aggregation --no-maxpool-metrics
# 	# 	done
# 	# done
# done
