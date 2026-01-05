import torch
from abc import ABC, abstractmethod


class Sampleable(ABC):
    """
    Base class for all sampleable objects.
    """
    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

class Alpha(ABC):
    """
    Base class for all alphas.
    """
    @abstractmethod
    def __init__(self, alpha : float):
        self.alpha = alpha

    @abstractmethod
    def __call__(self, time: float) -> float:
        raise NotImplementedError

class Beta(ABC):
    """
    Base class for all betas.
    """
    @abstractmethod
    def __init__(self, beta : float):
        self.beta = beta

    @abstractmethod
    def __call__(self, time: float) -> float:
        raise NotImplementedError

class Flow(ABC):
    """
    Base class for all flows.
    """
    @abstractmethod
    def __init__(self, alpha : float, beta : float):
        self.alpha = alpha
        self.beta = beta

    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError