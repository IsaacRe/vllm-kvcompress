"""Sequence and its related classes."""
import copy
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import math
import numpy as np

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence
from vllm.kvcompress.block import LogicalKVBlock


class CompressibleSequence(Sequence):

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        compression_rate: float,
        min_compressible_len: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        super().__init__(
            seq_id, prompt, prompt_token_ids, block_size, eos_token_id, lora_request)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        # Mapping: Layery index -> KV head index -> block count
        self.head_block_count = np.zeros(
            (num_layers, num_kv_heads), dtype=np.int32
        )
        # Mapping: Layery index -> KV head index -> KV count
        self.head_kv_count = np.zeros(
            (num_layers, num_kv_heads), dtype=np.int32
        )
        self.block_count = 0
        self.kv_count = 0
        self._append_tokens_to_blocks(prompt_token_ids)
        # Target compression rate for KV compression
        self.compression_rate = compression_rate
        # Minimum sequence length before compression is active
        self.min_compressible_len = min_compressible_len

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        num_tokens = len(token_ids)
        self.kv_count += num_tokens * self.num_layers * self.num_kv_heads
        self.head_kv_count += num_tokens
        # If we just filled a block, update the block count to add tokens
        # that will be generated during the next decoding step.
        new_head_block_count = (
            (self.head_kv_count + self.block_size) // self.block_size
        )
        new_blocks = int((new_head_block_count - self.head_block_count).sum())
        self.block_count += new_blocks
        self.head_block_count = new_head_block_count

    def schedule_compression(self, compression_rate: Optional[float] = None) -> int:
        """Return the number of this sequence's blocks to be freed during
        the next compression iteration.
        """
        if compression_rate is None:
            compression_rate = self.compression_rate
        # Total KV count should not go below min_compressible_len * num_kv_heads
        compressible_token_count = self.data.get_len() - self.min_compressible_len
        if compressible_token_count <= 0:
            return 0
        uncompressed_kv_count = (
            compressible_token_count * self.num_layers * self.num_kv_heads
        )
        compressed_kv_count = (
            self.kv_count -
            self.min_compressible_len * self.num_layers * self.num_kv_heads
        )
        target_kv_count = math.ceil(uncompressed_kv_count * compression_rate)
        evict_kv_count = max(0, compressed_kv_count - target_kv_count)
        evict_block_count = evict_kv_count // self.block_size
        return evict_block_count

    def remove_trailing_blocks(self, freed_block_count: np.ndarray):
        """Update sequences logical block state to account for the last
        iteration of compression by removing evictd blocks from the end
        of each head.

        Args:
            freed_block_count: count of freed blocks for each kv head
        """
        assert (self.num_layers, self.num_kv_heads) == freed_block_count.shape
        # Add block for tokens that will be generated during the next decoding
        # step.
        new_head_block_count = self.head_block_count - freed_block_count + 1
        evicted_head_kv_count = self.head_kv_count
        
        new_blocks = int((new_head_block_count - self.head_block_count).sum())
        self.block_count += new_blocks
        self.head_block_count = new_head_block_count
