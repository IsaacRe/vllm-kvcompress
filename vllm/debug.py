import torch
import os

_MAX_SAVE_ITERS = 3


class Checkpointer:
    
    def __init__(self) -> None:
        self.base_dir = "."
        self.do_checkpoint = False
        self.checkpoint_counts = {}

    def torch_save(self, name: str, tensor: torch.Tensor) -> None:
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        if name not in self.checkpoint_counts:
            self.checkpoint_counts[name] = 0
        if self.checkpoint_counts[name] < _MAX_SAVE_ITERS:
            path = os.path.join(self.base_dir, f'{name}__{self.checkpoint_counts[name]}')
            torch.save(tensor.cpu(), path)
        self.checkpoint_counts[name] += 1


CHECKPOINTER = Checkpointer()
