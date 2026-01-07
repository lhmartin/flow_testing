from flow_testing.flow.base import Sampleable, Flow, Alpha, Beta
from torch.distributions import MultivariateNormal
import torch

class GaussianDistribution(Sampleable, torch.nn.Module):
    """
    Class that represents a sampleable Gaussian Distribution.
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

class R3Flow(Flow):
    def __init__(self, alpha : Alpha, beta : Beta, distribution : GaussianDistribution):
        super().__init__(alpha, beta)
        self.distribution = distribution

    @property
    def dim(self) -> int:
        return self.distribution.dim

    def noise_sample(self, sample: torch.Tensor, time : float) -> torch.Tensor:
        noise = self.distribution.sample(sample.shape[0])

        alpha = self.alpha(time)
        beta = self.beta(time)

        return alpha * noise + beta * sample


if __name__ == "__main__":
    from flow_testing.flow.base import LinearAlpha, LinearBeta
    
    alpha = LinearAlpha()
    beta = LinearBeta()
    distribution = GaussianDistribution(mean=torch.zeros(3), cov=torch.eye(3))
    flow = R3Flow(alpha, beta, distribution)
    print(flow.noise_sample(sample=torch.zeros(10, 3), time=torch.tensor(0.5)))