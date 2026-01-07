import torch
from abc import ABC, abstractmethod


class Sampleable(ABC):
    """
    Base class for all sampleable objects.
    """
    @abstractmethod
    def sample(self, n_samples: int, shape: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError


class Alpha(ABC):
    """
    Base class for all alphas.
    """
    @abstractmethod
    def __init__(self):
        # check that alpha is of the correct form:
        assert torch.allclose(
            self(torch.zeros(1)),
            torch.zeros(1),
        )
        assert torch.allclose(
            self(torch.ones(1)),
            torch.ones(1),
        )

    @abstractmethod
    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class Beta(ABC):
    """
    Base class for all betas.
    """
    @abstractmethod
    def __init__(self):
        # check that beta is of the correct form:
        assert torch.allclose(
            self(torch.zeros(1)),
            torch.ones(1),
        )
        assert torch.allclose(
            self(torch.ones(1)),
            torch.zeros(1),
        )

    @abstractmethod
    def __call__(self, time: float) -> float:
        raise NotImplementedError

    def dt(self, time : torch.Tensor) -> torch.Tensor:
        t = time.unsqueeze(1)

class LinearAlpha(Alpha):
    """
    Linear alpha.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        return time

class LinearBeta(Beta):
    """
    Linear beta.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        return 1 - time

class Flow(ABC):
    """
    Base class for all flows.
    """
    @abstractmethod
    def __init__(self, alpha : float, beta : float):
        self.alpha = alpha
        self.beta = beta

    @abstractmethod
    def noise_sample(self, sample: torch.Tensor, time: float) -> torch.Tensor:
        raise NotImplementedError