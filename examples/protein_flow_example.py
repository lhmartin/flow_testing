"""
Example script demonstrating rotation and translation flow applied to a protein.

This script:
1. Loads a protein from a PDB file
2. Extracts backbone rigid frames (rotation + translation)
3. Applies R3 flow to translations (interpolates with Gaussian noise)
4. Applies SO(3) flow to rotations (interpolates with random rotations)
5. Reconstructs backbone at 200 timesteps and saves as a single trajectory
"""

import numpy as np
import torch
from pathlib import Path

from biotite.structure import AtomArray, stack
from biotite.structure.io.pdb import PDBFile

from flow_testing.data.protein import Protein
from flow_testing.data.rigid import Rigid
from flow_testing.data.rot import Rotation
from flow_testing.data.utils import calculate_backbone
from flow_testing.flow.base import LinearAlpha, LinearBeta
from flow_testing.flow.r3_flow import R3Flow, GaussianDistribution

# Number of timesteps for the flow trajectory
NUM_TIMESTEPS = 200


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

    def sample_noise(self, n: int) -> np.ndarray:
        """Sample random rotation matrices as noise target."""
        return random_rotation_matrices(n)

    def interpolate(self, rotation: Rotation, noise_rotations: np.ndarray, time: float) -> Rotation:
        """
        Interpolate between data rotations and pre-sampled noise rotations.

        Parameters:
        -----------
        rotation : Rotation
            The original rotation matrices
        noise_rotations : np.ndarray
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

        # Geodesic interpolation on SO(3)
        interpolated = so3_geodesic_interpolation(
            rotation.rot_mats,  # start (data)
            noise_rotations,    # end (noise)
            alpha_t             # interpolation parameter
        )

        return Rotation(interpolated)

    def noise_sample(self, rotation: Rotation, time: float) -> Rotation:
        """
        Apply noise to rotation matrices using geodesic interpolation.
        (Convenience method that samples new noise each call)
        """
        n = rotation.rot_mats.shape[0]
        noise_rotations = self.sample_noise(n)
        return self.interpolate(rotation, noise_rotations, time)


def generate_flow_trajectory(protein: Protein, num_timesteps: int = NUM_TIMESTEPS) -> list[AtomArray]:
    """
    Generate a trajectory of protein structures along the flow from data to noise.

    Parameters:
    -----------
    protein : Protein
        Input protein structure
    num_timesteps : int
        Number of timesteps in the trajectory

    Returns:
    --------
    list[AtomArray]
        List of AtomArray structures at each timestep
    """
    # Center the protein
    protein.center(type='backbone')

    # Extract backbone rigid frames and psi angles
    bb_rigid = protein.to_bb_rigid(center=False)
    psi_sin_cos = protein.to_psi_sin_cos()

    n_residues = bb_rigid.trans.shape[0]
    print(f"Protein has {n_residues} residues")
    print(f"Original translation range: [{bb_rigid.trans.min():.2f}, {bb_rigid.trans.max():.2f}]")

    # === Setup flows ===
    alpha = LinearAlpha()
    beta = LinearBeta()
    distribution = GaussianDistribution(
        mean=torch.zeros(3),
        cov=torch.eye(3) * 10.0  # Scale covariance for protein-sized structures
    )
    r3_flow = R3Flow(alpha, beta, distribution)
    so3_flow = SO3Flow()

    # === Pre-sample noise (same noise target for all timesteps for smooth trajectory) ===
    trans_tensor = torch.tensor(bb_rigid.trans, dtype=torch.float32)
    trans_noise = distribution.sample(n_residues)  # Sample translation noise once
    rot_noise = so3_flow.sample_noise(n_residues)  # Sample rotation noise once

    print(f"Generating {num_timesteps} timesteps...")

    # Generate frames at each timestep
    frames = []
    time_steps = np.linspace(0.0, 1.0, num_timesteps)

    for i, t in enumerate(time_steps):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing timestep {i + 1}/{num_timesteps} (t={t:.3f})")

        # Interpolate translations: x_t = beta(t) * x_data + alpha(t) * noise
        t_tensor = torch.tensor([t])
        alpha_t = alpha(t_tensor)
        beta_t = beta(t_tensor)
        noised_trans = beta_t * trans_tensor + alpha_t * trans_noise
        noised_trans_np = noised_trans.detach().numpy()

        # Interpolate rotations using geodesic interpolation
        noised_rot = so3_flow.interpolate(bb_rigid.rot, rot_noise, t)

        # Create noised rigid transformation
        noised_rigid = Rigid(noised_trans_np, noised_rot)

        # Reconstruct backbone from noised frames
        noised_backbone = calculate_backbone(noised_rigid, psi_sin_cos)

        # Create protein and convert to AtomArray
        noised_protein = Protein.from_backbone(noised_backbone)
        frames.append(noised_protein.to_biotite())

    return frames


def save_trajectory(frames: list[AtomArray], output_path: Path) -> None:
    """
    Save a list of AtomArray frames as a multi-model PDB trajectory file.

    Parameters:
    -----------
    frames : list[AtomArray]
        List of AtomArray structures to save
    output_path : Path
        Path to output PDB file
    """
    # Stack frames into AtomArrayStack
    trajectory = stack(frames)

    # Save using PDBFile
    pdb_file = PDBFile()
    pdb_file.set_structure(trajectory)
    pdb_file.write(str(output_path))


def main():
    """Main function demonstrating the flow matching pipeline."""

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

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

    print("\n" + "="*60)
    print(f"Generating Flow Trajectory with {NUM_TIMESTEPS} Timesteps")
    print("="*60 + "\n")

    # Generate trajectory
    frames = generate_flow_trajectory(protein, NUM_TIMESTEPS)

    # Save as single trajectory file
    trajectory_output = output_dir / "flow_trajectory.pdb"
    print(f"\nSaving trajectory to: {trajectory_output}")
    save_trajectory(frames, trajectory_output)

    print("\n" + "="*60)
    print("Flow Trajectory Complete!")
    print("="*60)
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - original_centered.pdb: Original protein structure")
    print(f"  - flow_trajectory.pdb: {NUM_TIMESTEPS}-frame trajectory (t=0 to t=1)")
    print("\nTo visualize the trajectory in PyMOL:")
    print("  1. Open flow_trajectory.pdb")
    print("  2. Use the playback controls to animate through frames")
    print("  3. Or use: cmd.mplay() to start animation")


if __name__ == "__main__":
    main()
