"""
Example script demonstrating rotation and translation flow applied to a protein.

This script:
1. Loads a protein from a PDB file
2. Extracts backbone rigid frames (rotation + translation)
3. Applies R3 flow to translations (interpolates with Gaussian noise)
4. Applies SO(3) flow to rotations (interpolates with random rotations)
5. Reconstructs and saves the noised protein backbone
"""

import numpy as np
import torch
from pathlib import Path

from flow_testing.data.protein import Protein
from flow_testing.data.rigid import Rigid
from flow_testing.data.rot import Rotation
from flow_testing.data.utils import calculate_backbone
from flow_testing.flow.base import LinearAlpha, LinearBeta
from flow_testing.flow.r3_flow import R3Flow, GaussianDistribution


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


class SO3Flow:
    """
    Simple SO(3) flow that interpolates between data rotations and random rotations.

    At t=0: returns the original data rotation
    At t=1: returns random rotation (noise)
    """

    def __init__(self):
        self.alpha = LinearAlpha()
        self.beta = LinearBeta()

    def noise_sample(self, rotation: Rotation, time: float) -> Rotation:
        """
        Apply noise to rotation matrices using geodesic interpolation.

        Parameters:
        -----------
        rotation : Rotation
            The original rotation matrices
        time : float
            Time parameter in [0, 1]. At t=0, returns original; at t=1, returns noise.

        Returns:
        --------
        Rotation
            Noised rotation matrices
        """
        n = rotation.rot_mats.shape[0]

        # Sample random rotations as noise
        noise_rotations = random_rotation_matrices(n)

        # Interpolate: at t=0 we want original, at t=1 we want noise
        # Using alpha(t) for noise weight and beta(t) for data weight
        t_tensor = torch.tensor([time])
        alpha_t = float(self.alpha(t_tensor))

        # Geodesic interpolation on SO(3)
        interpolated = so3_geodesic_interpolation(
            rotation.rot_mats,  # start (data)
            noise_rotations,    # end (noise)
            alpha_t             # interpolation parameter
        )

        return Rotation(interpolated)


def apply_flow_to_protein(protein: Protein, time: float, output_dir: Path) -> Protein:
    """
    Apply rotation and translation flow to a protein structure.

    Parameters:
    -----------
    protein : Protein
        Input protein structure
    time : float
        Flow time parameter in [0, 1]
    output_dir : Path
        Directory to save output files

    Returns:
    --------
    Protein
        Protein with noised backbone coordinates
    """
    # Center the protein
    protein.center(type='backbone')

    # Extract backbone rigid frames and psi angles
    bb_rigid = protein.to_bb_rigid(center=False)
    psi_sin_cos = protein.to_psi_sin_cos()

    n_residues = bb_rigid.trans.shape[0]
    print(f"  Protein has {n_residues} residues")
    print(f"  Original translation range: [{bb_rigid.trans.min():.2f}, {bb_rigid.trans.max():.2f}]")

    # === Apply R3 Flow to Translations ===
    # Create R3 flow with standard Gaussian distribution
    alpha = LinearAlpha()
    beta = LinearBeta()
    distribution = GaussianDistribution(
        mean=torch.zeros(3),
        cov=torch.eye(3) * 10.0  # Scale covariance for protein-sized structures
    )
    r3_flow = R3Flow(alpha, beta, distribution)

    # Convert translations to torch tensor and apply flow
    trans_tensor = torch.tensor(bb_rigid.trans, dtype=torch.float32)
    noised_trans = r3_flow.noise_sample(trans_tensor, time)
    noised_trans_np = noised_trans.detach().numpy()

    print(f"  Noised translation range: [{noised_trans_np.min():.2f}, {noised_trans_np.max():.2f}]")

    # === Apply SO(3) Flow to Rotations ===
    so3_flow = SO3Flow()
    noised_rot = so3_flow.noise_sample(bb_rigid.rot, time)

    # Create noised rigid transformation
    noised_rigid = Rigid(noised_trans_np, noised_rot)

    # Reconstruct backbone from noised frames
    noised_backbone = calculate_backbone(noised_rigid, psi_sin_cos)

    # Create protein from noised backbone
    noised_protein = Protein.from_backbone(noised_backbone)

    return noised_protein


def main():
    """Main function demonstrating the flow matching pipeline."""

    # Setup paths
    project_root = Path(__file__).parent.parent
    test_data_dir = project_root / "test-data" / "pdbs"
    output_dir = project_root / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load example protein
    pdb_path = test_data_dir / "8UVY.pdb"
    print(f"Loading protein from: {pdb_path}")
    protein = Protein.from_pdb(str(pdb_path))

    # Save original centered protein for comparison
    protein.center(type='backbone')
    original_output = output_dir / "original_centered.pdb"
    protein.to_pdb(str(original_output))
    print(f"Saved original centered protein to: {original_output}")

    # Apply flow at different time steps
    time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n" + "="*60)
    print("Applying Rotation and Translation Flow at Different Times")
    print("="*60)

    for t in time_steps:
        print(f"\nTime t={t}:")

        # Reload protein fresh for each time step
        protein = Protein.from_pdb(str(pdb_path))

        # Apply flow
        noised_protein = apply_flow_to_protein(protein, t, output_dir)

        # Save result
        output_path = output_dir / f"noised_t{t:.2f}.pdb"
        noised_protein.to_pdb(str(output_path))
        print(f"  Saved to: {output_path}")

    print("\n" + "="*60)
    print("Flow Example Complete!")
    print("="*60)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nTo visualize the flow progression, open the PDB files in PyMOL or similar:")
    print("  - original_centered.pdb: Original protein structure")
    print("  - noised_t0.00.pdb: No noise (should match original backbone)")
    print("  - noised_t0.25.pdb: 25% interpolation toward noise")
    print("  - noised_t0.50.pdb: 50% interpolation toward noise")
    print("  - noised_t0.75.pdb: 75% interpolation toward noise")
    print("  - noised_t1.00.pdb: Full noise (random structure)")


if __name__ == "__main__":
    main()
