from flow_testing.flow.base import Sampleable, Flow, Alpha, Beta
from torch.distributions import MultivariateNormal
import torch

class GaussianDistribution(Sampleable, torch.nn.Module):
    """
    Class that represents a Gaussian Distribution.
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        Parameters:
        -----------
        mean : torch.Tensor
            The mean of the Gaussian distribution.
            Expected shape: [dim]
        cov : torch.Tensor
            The covariance matrix of the Gaussian distribution.
            Expected shape: [dim, dim]
        """
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.distribution.sample((n_samples,))

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self) -> MultivariateNormal:
        return MultivariateNormal(self.mean, self.cov)

class R3Flow(Sampleable, Flow):
    def __init__(self, alpha : Alpha, beta : Beta):
        super().__init__(alpha, beta)

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.alpha(self.beta.sample(n_samples))