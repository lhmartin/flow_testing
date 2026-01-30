"""SO(3) rotation flow for interpolating between data and noise rotations."""

import torch
import numpy as np

from flow_testing.flow.base import Flow, Alpha, Beta, LinearAlpha, LinearBeta
from flow_testing.data.rot import Rotation
from flow_testing.flow.rotation_utils import SO3Operations, GeodesicInterpolant


class SO3Flow(Flow):
    """
    SO(3) flow that interpolates between data rotations and random rotations.

    At t=0: returns the original data rotation
    At t=1: returns random rotation (noise)

    This implements geodesic interpolation on the SO(3) manifold using
    the exp/log maps and Rodrigues' formula via rotation_utils.
    """

    def __init__(self, alpha: Alpha = None, beta: Beta = None):
        """
        Initialize SO3Flow.

        Parameters:
        -----------
        alpha : Alpha, optional
            Alpha schedule (defaults to LinearAlpha)
        beta : Beta, optional
            Beta schedule (defaults to LinearBeta)
        """
        if alpha is None:
            alpha = LinearAlpha()
        if beta is None:
            beta = LinearBeta()
        self.alpha = alpha
        self.beta = beta
        self._interpolant = GeodesicInterpolant()

    def sample_noise(self, n: int, device: torch.device = None,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Sample random rotation matrices as noise target.

        Parameters:
        -----------
        n : int
            Number of rotation matrices to sample
        device : torch.device, optional
            Device for the tensors
        dtype : torch.dtype, optional
            Data type for the tensors

        Returns:
        --------
        torch.Tensor
            Tensor of shape (n, 3, 3) containing random rotation matrices
        """
        return SO3Operations.sample_uniform(n, device=device, dtype=dtype)

    def interpolate(self, rotation: Rotation, noise_rotations: torch.Tensor, time: float) -> Rotation:
        """
        Interpolate between data rotations and pre-sampled noise rotations.

        Parameters:
        -----------
        rotation : Rotation
            The original rotation matrices
        noise_rotations : torch.Tensor
            Pre-sampled noise rotation matrices, shape (n, 3, 3)
        time : float
            Time parameter in [0, 1]. At t=0, returns original; at t=1, returns noise.

        Returns:
        --------
        Rotation
            Interpolated rotation matrices
        """
        t_tensor = torch.tensor([time])
        alpha_t = float(self.alpha(t_tensor))

        # Convert rotation matrices to torch tensors
        r0 = torch.tensor(rotation.rot_mats, dtype=noise_rotations.dtype)
        r1 = noise_rotations

        # Geodesic interpolation on SO(3) using rotation_utils
        interpolated = self._interpolant.interpolate(r0, r1, alpha_t)

        return Rotation(interpolated.numpy())

    def noise_sample(self, sample: torch.Tensor, time: float) -> Rotation:
        """
        Apply noise to rotation matrices using geodesic interpolation.

        Parameters:
        -----------
        sample : torch.Tensor or Rotation
            Input rotation (can be Rotation object or tensor of shape (n, 3, 3))
        time : float
            Time parameter in [0, 1]

        Returns:
        --------
        Rotation
            Noised rotation matrices
        """
        if isinstance(sample, Rotation):
            rotation = sample
        else:
            # Assume sample is a tensor of rotation matrices
            rotation = Rotation(sample.numpy() if isinstance(sample, torch.Tensor) else sample)

        n = rotation.rot_mats.shape[0]
        noise_rotations = self.sample_noise(n)
        return self.interpolate(rotation, noise_rotations, time)
