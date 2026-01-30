from flow_testing.flow.base import Alpha, Flow, Beta, Sampleable
from flow_testing.flow.rotation_utils import GeodesicInterpolant, SO3Distribution
import torch

class RotationDistribution(Sampleable):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.cov = cov

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.distribution.sample((n_samples,))

class RotFlow(Flow):
    def __init__(self, alpha : Alpha, beta : Beta):
        super().__init__(alpha, beta)

        self.distribution = SO3Distribution()
        self.interpolant = GeodesicInterpolant()

    @property
    def dim(self) -> int:
        return 3 # we are on SO(3)

    def noise_sample(self, sample: torch.Tensor, time : float) -> torch.Tensor:
        noise = self.distribution.sample(sample.shape[0], shape=(3, 3))

        r_t = self.interpolant.interpolate(sample, noise, time)

        return r_t

