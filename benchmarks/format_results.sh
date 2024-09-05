outfile=$1
echo 'model,input_len,output_len,max_cache_tokens,req_per_s,tok_per_s,max_batch_size,' > $outfile
for experiment in $(ls *.out); do
    split=(${experiment//./ })
    experiment_id=${split[0]}
    throughput=$(cat $experiment | grep Throughput | awk 'BEGIN { ORS="," };/^Throughput/{print $2}/requests/{print $4}')
    max_batch_size=$(cat $experiment | grep 'Max decoding batch' | awk 'BEGIN { ORS="," };/^Max decoding batch/{print $4}')
    if [ -z "$throughput" ]; then
        throughput=',,'
    fi
    if [ -z "$max_batch_size" ]; then
        max_batch_size=','
    fi
    echo ${experiment_id//_/,},$throughput$max_batch_size >> $outfile
done