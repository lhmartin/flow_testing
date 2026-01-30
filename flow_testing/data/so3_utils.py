"""SO(3) rotation utilities for rotation sampling and geodesic interpolation."""

import numpy as np


def random_rotation_matrices(n: int) -> np.ndarray:
    """
    Generate n random rotation matrices using QR decomposition.

    Parameters:
    -----------
    n : int
        Number of rotation matrices to generate

    Returns:
    --------
    np.ndarray
        Array of shape (n, 3, 3) containing random rotation matrices
    """
    # Generate random matrices
    random_matrices = np.random.randn(n, 3, 3)

    # Use QR decomposition to get orthogonal matrices
    rot_matrices = np.zeros((n, 3, 3))
    for i in range(n):
        q, r = np.linalg.qr(random_matrices[i])
        # Ensure proper rotation (det = 1, not -1)
        d = np.linalg.det(q)
        q[:, -1] *= d
        rot_matrices[i] = q

    return rot_matrices


def so3_geodesic_interpolation(rot_start: np.ndarray, rot_end: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolate between two rotation matrices along the geodesic on SO(3).
    Uses the formula: R(t) = R_start @ expm(t * logm(R_start.T @ R_end))

    For simplicity, we use a linear interpolation in the tangent space.

    Parameters:
    -----------
    rot_start : np.ndarray
        Starting rotation matrices, shape (n, 3, 3)
    rot_end : np.ndarray
        Ending rotation matrices, shape (n, 3, 3)
    t : float
        Interpolation parameter in [0, 1]

    Returns:
    --------
    np.ndarray
        Interpolated rotation matrices, shape (n, 3, 3)
    """
    n = rot_start.shape[0]
    result = np.zeros_like(rot_start)

    for i in range(n):
        # Compute relative rotation: R_rel = R_start.T @ R_end
        R_rel = rot_start[i].T @ rot_end[i]

        # Compute axis-angle representation
        # Use Rodrigues' formula to extract angle
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        if angle < 1e-6:
            # Very small rotation, just use start
            result[i] = rot_start[i]
        else:
            # Extract axis from skew-symmetric part
            skew = (R_rel - R_rel.T) / 2
            axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
            else:
                axis = np.array([1.0, 0.0, 0.0])

            # Interpolate angle
            interp_angle = t * angle

            # Rodrigues' formula for interpolated rotation
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_interp = np.eye(3) + np.sin(interp_angle) * K + (1 - np.cos(interp_angle)) * (K @ K)

            # Compose with start rotation
            result[i] = rot_start[i] @ R_interp

    return result
