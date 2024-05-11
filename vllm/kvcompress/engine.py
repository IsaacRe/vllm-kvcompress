from typing import List, Dict
from dataclasses import dataclass
import torch

from vllm.config import KVCompressConfig


@dataclass
class CompressionOutputs:
    """This class defined the physical KV cache slot moves and block frees
    that must occur before all other scheduled cache actions and model-running.
    """

    cache_moves: List[torch.Tensor]
    freed_block_count: List[Dict[int, torch.Tensor]]


class KVCompressEngine:
    """Main runtime that handles the KV-Compress cache
    eviction cycle.

    While KV-Compress engine is running, any modification to KV cache
    or block tables other than adding to free blocks is disallowed.
    Before evicting or preempting any sequences the generation process
    must either wait for the current compression cycle to complete, or
    cancel it.
    
    There are 5 main steps within each iteration:
        1. Load eviction metrics from each layer's volatile memory
            space (written to by the generation process) to the memory
            block of the KV-Compress process.
        1. Sort the eviction metric across all KVs in each sequence.
        2. Schedule KV evictions (indexed by virtual position) based
            on the sorted KV list of each sequence.
        3. Schedule key and value moves (indexed by physical cache
            position) based on the scheduled evictions of each sequence.
            This requires sorting the output list of KV evictions from
            the last step by ascending index.
        4. Apply cache moves from the previous step to key and value
            cache, as well as to each layer's runnign eviction metrics
            stored in volatile memory. This can be done on a per-layer
            basis and will require a lock on that layer's KV cache and
            metrics tensor while executing.
        5. Update block tables to mark newly freed blocks.

    The writeable metrics tensor of the main generation process is source
    of truth for the metrics used for compression scheduling.

    Main generation process has the following additional responsabilities:
        - Add to running eviction metrics after each attention operation.
        - Upon eviction of a block, set all KV metrics for that block to
            infinity.
        - Wait for current compression iteration to complete before
            performing any block eviction.
        - If it runs out of block space, cancel the current compression
            iteration before performing preemption.

    """

    def __init__(self, config: KVCompressConfig) -> None:
        self.config = config

    def _allocate_mem(self):
        """Allocate memory for:
            - Eviction metric for each KV ( num_layers * max_num_blocks * block_size * 2 )
                - Allocate double space for torch.sort
            - Block metadata: ( num_layers * max_num_blocks per tensor )
                - Layer index
                - KV head index
                - Virtual block number
            - expanded sequence indices per KV
            - seq_block_offsets
            - copy of context_lens
            - truncated context_lens
            - evicted_blocks_per_sequence
            - 
            - evicted_kv_count
            - truncated evicted_kv_count
            - evicted_kv_indices
            - sorted_evicted_kv_indices (plus torch.sort overhead)
            - 
            - cache_moves_idx
            - cache_moves_count
            - 
        """

    def _sort_eviction_metrics(self):
        pass

    def compress(self):
        # acquire lock on metrics and context_lens tensors and
        # copy to KVCompress memory (layerwise)

        # truncate context_lens down to the last full block

        # initialize block metadata

        # sort metrics by ascending (metric, sequence index)

        # compute block offsets in sorted tensor per sequence

        # run schedule_cache_evictions

        # truncate per-head eviction counts down to last evicted block

        # sort per-head eviction indices by ascending index
        
        # run schedule_cache_moves (layerwise)

        # acquire lock on k/v cache tensors and run
        # execute_cache_moves (layerwise)

        # notify generation process of newly freed blocks

        pass
