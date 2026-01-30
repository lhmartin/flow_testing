"""
FoldFlow-Base: Flow Matching on SO(3) for Rotation Distributions

Implementation based on the paper "SE(3) Stochastic Flow Matching for Protein Backbone Generation"
(arXiv:2310.02391), specifically Section 3.1 on FoldFlow-Base.

This module implements:
- SO(3) parametrization using orthogonal matrices with unit determinant
- Lie algebra so(3) operations (skew-symmetric matrices)
- Riemannian metric on SO(3): <r, r'>_{SO(3)} = tr(rr'^T)/2
- Geodesic interpolation on SO(3)
- Conditional vector fields for flow matching
- FoldFlow-Base loss computation
"""

import torch
import numpy as np
from flow_testing.flow.base import Sampleable, Beta, Alpha

class SO3Operations:
    """
    Core operations on the SO(3) Lie group and its Lie algebra so(3).
    
    SO(3) is the group of 3D rotations, represented as 3x3 orthogonal matrices
    with determinant +1. The Lie algebra so(3) consists of 3x3 skew-symmetric
    matrices, which are tangent vectors at the identity of SO(3).
    """
    
    @staticmethod
    def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """
        Convert a 3D vector to its skew-symmetric matrix representation.
        
        The skew-symmetric matrix [v]_× satisfies: [v]_× @ w = v × w (cross product)
        
        Args:
            v: Tensor of shape (..., 3) representing axis-angle vectors
            
        Returns:
            Skew-symmetric matrices of shape (..., 3, 3)
        """
        # v has shape (..., 3)
        batch_shape = v.shape[:-1]
        
        # Extract components
        v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
        
        # Build skew-symmetric matrix
        zeros = torch.zeros_like(v1)
        
        # [  0  -v3   v2 ]
        # [ v3    0  -v1 ]
        # [-v2   v1    0 ]
        skew = torch.stack([
            torch.stack([zeros, -v3, v2], dim=-1),
            torch.stack([v3, zeros, -v1], dim=-1),
            torch.stack([-v2, v1, zeros], dim=-1)
        ], dim=-2)
        
        return skew
    
    @staticmethod
    def unskew(skew: torch.Tensor) -> torch.Tensor:
        """
        Extract the 3D vector from a skew-symmetric matrix.
        
        Args:
            skew: Skew-symmetric matrices of shape (..., 3, 3)
            
        Returns:
            Vectors of shape (..., 3)
        """
        # Extract: v1 = skew[2,1], v2 = skew[0,2], v3 = skew[1,0]
        v1 = skew[..., 2, 1]
        v2 = skew[..., 0, 2]
        v3 = skew[..., 1, 0]
        
        return torch.stack([v1, v2, v3], dim=-1)
    
    @staticmethod
    def exp_map(omega: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Exponential map from so(3) to SO(3) using Rodrigues' formula.
        
        exp(omega) = I + sin(θ)/θ * [ω]_× + (1-cos(θ))/θ² * [ω]_×²
        
        where θ = ||ω|| and [ω]_× is the skew-symmetric matrix of ω.
        
        Args:
            omega: Axis-angle vectors of shape (..., 3)
            eps: Small value for numerical stability
            
        Returns:
            Rotation matrices of shape (..., 3, 3)
        """
        theta = torch.norm(omega, dim=-1, keepdim=True)  # (..., 1)
        theta_sq = theta ** 2
        
        # For small angles, use Taylor expansion
        # sin(θ)/θ ≈ 1 - θ²/6
        # (1-cos(θ))/θ² ≈ 1/2 - θ²/24
        small_angle = theta.squeeze(-1) < eps
        
        # Compute coefficients
        sin_theta_over_theta = torch.where(
            theta > eps,
            torch.sin(theta) / theta,
            1.0 - theta_sq / 6.0
        )
        
        one_minus_cos_over_theta_sq = torch.where(
            theta > eps,
            (1.0 - torch.cos(theta)) / theta_sq,
            0.5 - theta_sq / 24.0
        )
        
        # Get skew-symmetric matrix
        omega_skew = SO3Operations.skew_symmetric(omega)  # (..., 3, 3)
        omega_skew_sq = omega_skew @ omega_skew
        
        # Rodrigues formula
        eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
        eye = eye.expand(*omega.shape[:-1], 3, 3)
        
        R = (eye + 
             sin_theta_over_theta.unsqueeze(-1) * omega_skew + 
             one_minus_cos_over_theta_sq.unsqueeze(-1) * omega_skew_sq)
        
        return R
    
    @staticmethod
    def log_map(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Logarithm map from SO(3) to so(3).
        
        This computes the axis-angle representation of a rotation matrix.
        log(R) returns an element of so(3) (a 3D vector).
        
        Args:
            R: Rotation matrices of shape (..., 3, 3)
            eps: Small value for numerical stability
            
        Returns:
            Axis-angle vectors of shape (..., 3)
        """
        batch_shape = R.shape[:-2]
        
        # Compute rotation angle from trace
        # tr(R) = 1 + 2*cos(θ)
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_theta)  # (...,)
        
        # For small angles: (R - R^T) / 2 ≈ [ω]_×
        # For larger angles: [ω]_× = θ/(2*sin(θ)) * (R - R^T)
        
        sin_theta = torch.sin(theta)
        
        # Handle different cases
        small_angle = theta < eps
        near_pi = theta > (np.pi - eps)
        
        # Standard case coefficient
        coeff = torch.where(
            sin_theta.abs() > eps,
            theta / (2.0 * sin_theta),
            1.0 + theta ** 2 / 6.0  # Taylor expansion
        )
        
        # Skew-symmetric part
        skew = (R - R.transpose(-2, -1)) / 2.0
        omega = SO3Operations.unskew(skew) * coeff.unsqueeze(-1)
        
        # Handle θ ≈ π case (sin(θ) ≈ 0)
        # In this case, use: R = I + 2*sin²(θ/2)/θ² * [ω]_×² = I + 2*[n]_×²
        # where n is the unit axis, so R + I has rank 1 with eigenvector n
        if near_pi.any():
            # For near-π rotations, extract axis from R + I
            R_plus_I = R + torch.eye(3, device=R.device, dtype=R.dtype)
            # The column with largest norm gives the axis
            norms = torch.norm(R_plus_I, dim=-2)
            max_idx = torch.argmax(norms, dim=-1)
            
            # This is a simplified handling - for production, would need more care
            axis = R_plus_I[..., :, max_idx.unsqueeze(-1).expand(*batch_shape, 3, 1)]
            axis = axis.squeeze(-1)
            axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + eps)
            omega_near_pi = axis * theta.unsqueeze(-1)
            
            omega = torch.where(near_pi.unsqueeze(-1), omega_near_pi, omega)
        
        return omega
    
    @staticmethod
    def log_map_relative(R1: torch.Tensor, R0: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Compute log_{R0}(R1) - the logarithm map at R0 evaluated at R1.
        
        This gives the tangent vector at R0 pointing towards R1.
        log_{R0}(R1) lives in the tangent space T_{R0}SO(3).
        
        Mathematically: log_{R0}(R1) = R0 @ log(R0^T @ R1)
        
        The result is a skew-symmetric matrix in T_{R0}SO(3).
        We return it as a 3D vector (the axis-angle representation).
        
        Args:
            R1: Target rotation matrices of shape (..., 3, 3)
            R0: Base point rotation matrices of shape (..., 3, 3)
            eps: Small value for numerical stability
            
        Returns:
            Tangent vectors at R0 of shape (..., 3)
        """
        # Compute relative rotation
        R_rel = R0.transpose(-2, -1) @ R1  # R0^T @ R1
        
        # Log at identity
        omega_at_identity = SO3Operations.log_map(R_rel, eps)
        
        # The tangent vector at R0 (as axis-angle in the tangent space)
        # Since we parallel transport using left multiplication, we return omega directly
        # The skew-symmetric matrix form would be: R0 @ skew(omega)
        return omega_at_identity
    
    @staticmethod
    def so3_metric(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Compute the bi-invariant Riemannian metric on SO(3).
        
        <v1, v2>_{SO(3)} = tr(v1 @ v2^T) / 2
        
        For vectors in axis-angle form, this is equivalent to the dot product.
        
        Args:
            v1, v2: Tangent vectors of shape (..., 3)
            
        Returns:
            Inner products of shape (...)
        """
        return (v1 * v2).sum(dim=-1) / 2.0
    
    @staticmethod
    def so3_norm_squared(v: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared norm under the SO(3) metric.
        
        ||v||²_{SO(3)} = tr(v @ v^T) / 2 = ||v||² / 2
        
        Args:
            v: Tangent vectors of shape (..., 3)
            
        Returns:
            Squared norms of shape (...)
        """
        return (v ** 2).sum(dim=-1) / 2.0
    
    @staticmethod
    def sample_uniform(batch_size: int, device: torch.device = None, 
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Sample uniformly from SO(3) using the Haar measure.
        
        Uses the QR decomposition method for uniform sampling.
        
        Args:
            batch_size: Number of samples
            device: Torch device
            dtype: Data type
            
        Returns:
            Rotation matrices of shape (batch_size, 3, 3)
        """
        # Sample from standard normal
        Z = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
        
        # QR decomposition
        Q, R = torch.linalg.qr(Z)
        
        # Ensure proper rotation (det = +1)
        diag_sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        Q = Q * diag_sign.unsqueeze(-2)
        
        # Fix determinant if needed
        det = torch.det(Q)
        Q[det < 0, :, 0] *= -1
        
        return Q


class SO3Distribution(Sampleable):
    """
    Class that represents a distribution on SO(3).
    """
    def __init__(self):
        super().__init__()
    
    def sample(self, n_samples: int, shape: tuple[int, ...]) -> torch.Tensor:
        return SO3Operations.sample_uniform(n_samples)

class GeodesicInterpolant:
    """
    Geodesic interpolation on SO(3) for flow matching.
    
    Given r0 and r1 in SO(3), the geodesic interpolant is:
        r_t = exp_{r0}(t * log_{r0}(r1))
    
    This traces the shortest path (geodesic) from r0 to r1 on SO(3).
    """
    
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.so3_ops = SO3Operations()
    
    def interpolate(self, r0: torch.Tensor, r1: torch.Tensor, 
                    t: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic interpolant r_t between r0 and r1.
        
        r_t = exp_{r0}(t * log_{r0}(r1))
        
        Using the numerical trick: we compute log at identity and parallel transport.
        
        Args:
            r0: Source rotations of shape (..., 3, 3)
            r1: Target rotations of shape (..., 3, 3)
            t: Interpolation parameter in [0, 1], shape (...,) or scalar
            
        Returns:
            Interpolated rotations r_t of shape (..., 3, 3)
        """
        # Compute relative rotation: r0^T @ r1
        r_rel = r0.transpose(-2, -1) @ r1
        
        # Log at identity (axis-angle of relative rotation)
        omega = SO3Operations.log_map(r_rel, self.eps)  # (..., 3)
        
        # Scale by t
        if isinstance(t, (int, float)):
            omega_t = omega * t
        else:
            # Handle broadcasting
            while t.dim() < omega.dim():
                t = t.unsqueeze(-1)
            omega_t = omega * t
        
        # Exponential map to get the interpolated relative rotation
        r_rel_t = SO3Operations.exp_map(omega_t)
        
        # Apply to r0 to get r_t = r0 @ exp(t * log(r0^T @ r1))
        r_t = r0 @ r_rel_t
        
        return r_t
    
    def compute_velocity(self, r0: torch.Tensor, r1: torch.Tensor,
                         r_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional velocity field u_t(r_t | z) where z = (r0, r1).
        
        The velocity is computed as: u_t = log_{r_t}(r0) / t
        
        This is the constant velocity vector field that transports r0 to r1.
        Actually, we want the velocity pointing from r_t towards r1 (forward direction).
        
        Following the paper: u_t = log_{r_t}(r1) / (1 - t) for forward
        Or equivalently for training: target = log_{r_t}(r0) / t (pointing backward)
        
        Args:
            r0: Source rotations of shape (..., 3, 3)
            r1: Target rotations of shape (..., 3, 3)  
            r_t: Current interpolated rotations of shape (..., 3, 3)
            t: Time parameter in (0, 1), shape (...,) or scalar
            
        Returns:
            Velocity vectors in T_{r_t}SO(3), shape (..., 3)
        """
        # Compute log_{r_t}(r0) - tangent vector at r_t pointing towards r0
        # r_t^T @ r0 gives relative rotation
        r_rel = r_t.transpose(-2, -1) @ r0
        omega = SO3Operations.log_map(r_rel, self.eps)  # log at identity
        
        # Divide by t to get constant velocity
        # Handle t near 0
        if isinstance(t, (int, float)):
            t_safe = max(t, self.eps)
            velocity = omega / t_safe
        else:
            while t.dim() < omega.dim():
                t = t.unsqueeze(-1)
            t_safe = torch.clamp(t, min=self.eps)
            velocity = omega / t_safe
        
        return velocity