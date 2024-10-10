from dataclasses import dataclass

from vllm.kvcompress.metrics import CompressionMetrics
from vllm.kvcompress.block import BlockState


@dataclass
class KVCompressState:
    block_state: BlockState
    kv_metrics: CompressionMetrics
