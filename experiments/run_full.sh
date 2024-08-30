for dataset in $(cat datasets.txt);
do
    python3 run_longbench.py --protected-window-size 32 --metric-collection-buffer-size 10 --prefill-metric-collection-window-size 32 --kv-head-bias-weight 0 --dataset $dataset --max-cache-tokens -1 --metric-aggregation L2-sum --no-maxpool-metrics
done