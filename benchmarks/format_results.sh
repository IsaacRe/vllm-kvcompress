outfile=$1
echo 'model,input_len,output_len,max_cache_tokens,req_per_s,tok_per_s,' > $outfile
for experiment in $(ls *.out); do
    split=(${experiment//./ })
    experiment_id=${split[0]}
    throughput=$(cat $experiment | grep Throughput | awk 'BEGIN { ORS="," };/^Throughput/{print $2}/requests/{print $4}')
    if [ -z "$throughput" ]; then
        throughput=',,'
    fi
    echo ${experiment_id//_/,},$throughput >> $outfile
done