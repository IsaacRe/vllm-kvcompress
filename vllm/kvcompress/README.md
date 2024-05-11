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

### Open Questions
Re-adjust watermark level
Add on-demand compression (to avoid preemption/swapping)
How to support multi-GPU? (broadcast full block tables across all devices)
How to support chunked-prefill?
How to support prefix-caching/beam-search? -- will NOT support
Support for CPU/neuron implementations