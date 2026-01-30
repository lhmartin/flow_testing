import numpy as np
from flow_testing.data.rigid import Rigid
from flow_testing.data.protein_constants import (
    idealized_AA_positions, restype_1to3, restypes, DefaultFrames
)
from flow_testing.data.rot import Rotation


class BackboneBuilder:
    """Builder class for constructing backbone coordinates from rigid frames."""

    @staticmethod
    def psi_angles_to_rotation(psi_angles: np.ndarray) -> Rotation:
        """
        Convert psi angles (sin/cos form) to rotation matrices.

        Parameters:
        -----------
        psi_angles : np.ndarray
            Array of shape [n, 2] containing [sin(psi), cos(psi)] for each residue

        Returns:
        --------
        Rotation
            Rotation object containing the psi rotation matrices
        """
        template = np.zeros((psi_angles.shape[0], 3, 3))

        template[:, 0, 0] = 1
        template[:, 1, 1] = psi_angles[:, 1]
        template[:, 1, 2] = -psi_angles[:, 0]
        template[:, 2, 1] = psi_angles[:, 0]
        template[:, 2, 2] = psi_angles[:, 1]

        return Rotation(template)

    @staticmethod
    def calculate_backbone(bb_rigid: Rigid, psi_angles: np.ndarray) -> np.ndarray:
        """
        Calculate the 3D positions of backbone atoms given rigid frames and psi angles.

        Parameters:
        -----------
        bb_rigid : Rigid
            Backbone rigid frames
        psi_angles : np.ndarray
            Array of shape [n, 2] containing [sin(psi), cos(psi)] for each residue

        Returns:
        --------
        np.ndarray
            Backbone atom positions of shape [n, 5, 3]
        """
        n_residues = psi_angles.shape[0]
        ideal_amino_acid = idealized_AA_positions[restype_1to3[restypes[0]]]
        bb_atoms_ideal = np.stack([list(ideal_amino_acid[x][2]) for x in range(5)])

        # create a template array of the idealized positions
        backbone = np.zeros((n_residues, 5, 3))

        # apply the rigid frame to the idealized positions for first 4 atoms
        for atom_idx in range(4):
            ideal_pos = np.tile(bb_atoms_ideal[atom_idx], (n_residues, 1))
            backbone[:, atom_idx, :] = bb_rigid.apply(ideal_pos)

        # Default frame 3 transformation for the psi group
        default_rot = DefaultFrames.PSI_ROT
        default_trans = DefaultFrames.PSI_TRANS

        default_psi_rigid = Rigid(
            np.tile(default_trans, (n_residues, 1)),
            Rotation(np.tile(default_rot[None, :, :], (n_residues, 1, 1)))
        )

        # Create psi rotation from torsion angles
        psi_rotation = BackboneBuilder.psi_angles_to_rotation(psi_angles)
        psi_rot_rigid = Rigid(np.zeros((n_residues, 3)), psi_rotation)

        # Compose: default_frame -> psi_rotation
        psi_frame = default_psi_rigid.compose(psi_rot_rigid)  # psi_local -> bb_local
        combined_rigid = bb_rigid.compose(psi_frame)  # psi_local -> global
        o_ideal = np.tile(bb_atoms_ideal[4], (n_residues, 1))
        backbone[:, 4, :] = combined_rigid.apply(o_ideal)

        return backbone


# Backward compatibility aliases
def psi_angles_to_rotation(psi_angles: np.ndarray) -> Rotation:
    """Deprecated: Use BackboneBuilder.psi_angles_to_rotation instead."""
    return BackboneBuilder.psi_angles_to_rotation(psi_angles)


def calculate_backbone(bb_rigid: Rigid, psi_angles: np.ndarray) -> np.ndarray:
    """Deprecated: Use BackboneBuilder.calculate_backbone instead."""
    return BackboneBuilder.calculate_backbone(bb_rigid, psi_angles)
