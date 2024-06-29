import torch
import os
from typing import Optional

_MAX_SAVE_ITERS = 3
_MANIFEST_FILENAME = "files.txt"


class Checkpointer:
    
    def __init__(self) -> None:
        self.base_dir = "."
        self.do_save = False
        self.do_validate = False
        self.checkpoint_counts = {}
        self.files = []
        self.manifest = None
        self.disabled = False
        self.conditions = {}

    def __del__(self) -> None:
        if self.manifest is not None:
            self.manifest.close()

    def set_condition(self, **conditions) -> None:
        self.conditions.update(conditions)

    def init_manifest(self) -> None:
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.manifest = open(os.path.join(self.base_dir, _MANIFEST_FILENAME), "w+")

    @property
    def do_checkpoint(self) -> bool:
        return self.do_save or self.do_validate

    def torch_save(self, name: str, tensor: torch.Tensor) -> None:
        path = os.path.join(self.base_dir, f'{name}__{self.checkpoint_counts[name]}')
        torch.save(tensor.cpu(), path)

    def torch_load(self, name: str) -> torch.Tensor:
        path = os.path.join(self.base_dir, f'{name}__{self.checkpoint_counts[name]}')
        if not os.path.exists(path):
            raise RuntimeError('file not found: {path}')
        return torch.load(path)
    
    def checkpoint(self, name: str, tensor: torch.Tensor, max_save_iters: Optional[int] = None):
        if max_save_iters is None:
            max_save_iters = _MAX_SAVE_ITERS
        if not self.do_checkpoint:
            return
        if name not in self.checkpoint_counts:
            if self.manifest is None:
                self.init_manifest()
            self.checkpoint_counts[name] = 0
            self.files.append(name)
            self.manifest.write(name)
            self.manifest.flush()
        if self.checkpoint_counts[name] < max_save_iters:
            if self.do_save:
                self.torch_save(name, tensor)
            elif self.do_validate:
                reference = self.torch_load(name)
                assert (reference.to(tensor.device) == tensor).all(), f'checkpoint {name}({self.checkpoint_counts[name]}) failed:\n{reference} != {tensor}'
                print(f'checkpoint {name}({self.checkpoint_counts[name]}) passed')
        self.checkpoint_counts[name] += 1

    def condition(self, **condition_params) -> None:
        """Disables checkpointing if conditions are not met until
        next call to end_condition().
        """
        self.disabled = True
        for name, value in condition_params.items():
            if name not in self.conditions:
                raise RuntimeError(f'undefined condition {name}')
            if self.conditions[name](value):
                self.disabled = False
                return

    def end_condition(self):
        self.disabled = False


CHECKPOINTER = Checkpointer()
