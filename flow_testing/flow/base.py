import torch
from abc import ABC, abstractmethod


class Sampleable(ABC):
    """
    Base class for all sampleable objects.
    """
    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError