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
Transpose context_lens input to schedule_cache_evictions kernel
After benchmarking, move to main branch and remove t1/t2 block table references (just use t1)
Fix seq vs batch block state views (just have everything be a view)
Clean up block_state/block_manager/block_allocator responsabilities (probably only need two classes)
Passing of block_tables/context_lens/slot_mapping between block state and model runner is a mess
Remove context_lens transpose before schedule_evictions and remove contiguous calls
remove int conversions
Test generation after evicting due to completion/OOM
Remove arguments from LLM config that are now passed in SampleParams


### Open Questions
Re-adjust watermark level
Add on-demand compression (to avoid preemption/swapping) (use max_cache_tokens)
How to support multi-GPU? (broadcast full block tables across all devices)
How to support chunked-prefill?
How to support prefix-caching/beam-search?
Support for alibi positional encodings
Support for CPU/neuron implementations