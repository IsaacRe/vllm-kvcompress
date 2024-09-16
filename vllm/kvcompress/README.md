## Integration Notes

### Synchronous
- LLMEngine.step()
    - Scheduler.schedule()
        - KVCompressScheduler.run() -> KVCompressSchedulerOutput
            - determine how many evictions per sequence for this iteration
        - return (SchedulerOutput, KVCompressSchedulerOutput)
    - KVCompressEngine.run(metrics reference, KVCompressSchedulerOutput) -> cache_moves
        - if compression is scheduled for this iteration, execute an itertation of cache compression to get cache moves
    - ModelExecutor.worker.model_runner.run()
    - ModelExecutor.worker.cache_engine.execute_cache_moves(cache_moves)
        - cache_engine executes the cache moves after the model_runner is called (must be after since block_table references in SequenceGroupMetadata will be incorrect after moves are applied)


### to-do
Kernel for reshape_and_cache
benchmark kernels
test new logic/modify tests for existing logic
slot_mapping stuff
Modify prefill logic? model-runner/paged attention/other attn backends python logic
Rewrite block manager to use pre-allocated block tables
make sure position id only implicated in new token ids
make sure the block table accessed by the driver worker is equivalent to the updated block tables owned by the block_manager
Ok to not pass block tables during profiling phase? (They'll already be allocated but do we need them during dummy forward?)
Selective broadcast of block tables by block_manager to worker devices during scheduling/compression phase
Check for int32 overflows
Make everything use unified cache tensor for all layers (need int64 indices)
Make unified schedule_cache_moves kernel
Automate selection of execute_cache_moves CUDA threads/blocks
Initialize kv metrics during prefill
Modify paged attention kernel v2 kernel to reduce kv metrics in the same mem space as the output
KV metric reduction (pass through modified code)
Get kv head sporadicity and num_queries_per_kv to the CompressionMetric.aggregate method
Either profile mem for metric sorting/aggregation or allocate upfront
Cache move needs to move the kv metrics as well
Transpose context_lens input to schedule_cache_evictions kernel
Pass kv_metric_buffer_len to kvc attention kernel to filter which keys have their emtrics aggregated to
After benchmarking, move to main branch and remove t1/t2 block table references (just use t1)
Fix seq vs batch block state views (just have everything be a view)
Clean up block_state/block_manager/block_allocator responsabilities (probably only need two classes)
Passing of block_tables/context_lens/slot_mapping between block state and model runner is a mess
Remove increment_on_full, this will remove the need for move_empty_trailing_blocks
Remove context_lens transpose before schedule_evictions and remove contiguous calls
remove int conversions
Improve memory profiling - profile full kvc schedule loop (2nd sort is causing OOM)
Non-determinism in compression at previous commit (e3c9b5253b8022ed2d24b90a7cc26d5803629653) when running random digit compression test with 100 protected tokens -- yet deterministic when running with 50 protected tokens
Test generation after evicting due to completion/OOM
Agreggate temp_metrics after each layer's attention rather than at end
Is kv metadata getting moved during execute_cache_moves/does it need to be? - Just need to reorder metrics tensor--logical block numbers remain valid since both physical and logical blocks are changed when a KV is moved.
Need to skip KVs within protected window in schedule_evictions kernel and skip recording of attention between Qs and Ks within buffer length in paged attention kernel.
Test with multiple input prompts
Validate kvc paged attention v2 kernel correctness
Validate GQA correctness
Use evicted_kv_count in _schedule_compression()
Use stable=True during sort()
Get eviction test passing for even_layer_evict and protected_window > 0
Remove arguments from LLM config that are now passed in SampleParams
Test correctness of compress_once
Fix bug: encountering negative context length eventually when running large batch decoding
- reproduce with `python3 benchmarks/benchmark_throughput.py --num-prompts 256 --input-len 250 --output-len 500 --enforce-eager --model meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 19000 --max-cache-tokens 128 --new-token-limit 249 --kvc-interval 1000000 --metric-collection-buffer-size 10 --protected-window-size 32 --enable-kvc --max-kv-per-compression 5000000` on L4


### Potential Problem Code
remove_trailing_blocks and other code used to free evicted KV blocks and reorganize memory
- Test correctness after evicting sequence


### Open Questions
Re-adjust watermark level
Add on-demand compression (to avoid preemption/swapping) (use max_cache_tokens)
How to support multi-GPU? (broadcast full block tables across all devices)
How to support chunked-prefill?
How to support prefix-caching/beam-search? -- will NOT support
Support for alibi positional encodings
Support for CPU/neuron implementations