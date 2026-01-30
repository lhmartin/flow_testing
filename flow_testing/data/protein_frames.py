"""Frame calculation utilities for Protein structures."""

from typing import TYPE_CHECKING
import numpy as np

from flow_testing.data.rigid import Rigid, RigidBuilder
from flow_testing.data.rot import Rotation
from flow_testing.data.protein_constants import DefaultFrames

if TYPE_CHECKING:
    from flow_testing.data.protein import Protein


class ProteinFrameBuilder:
    """Builds rigid frames from protein structures."""

    @staticmethod
    def to_bb_rigid(protein: 'Protein', center: bool = True) -> Rigid:
        """
        Convert a protein to a Rigid object using the backbone atoms.
        Uses atoms ['C', 'CA', 'N'] to define the local frame.

        This goes from the local coordinate system of the backbone to the global coordinate system.

        Parameters:
        -----------
        protein : Protein
            Input protein structure
        center : bool
            Whether to center the coordinates first

        Returns:
        --------
        Rigid
            Backbone rigid frames
        """
        bb_atoms = protein.atom_positions[:, [2, 1, 0], :]
        if center:
            bb_atoms = bb_atoms - np.mean(bb_atoms, axis=0)

        rigid = RigidBuilder.from_matrix_local_to_global(bb_atoms)

        return rigid

    @staticmethod
    def to_psi_sin_cos(protein: 'Protein') -> np.ndarray:
        """
        Calculate the sin and cos of the psi angle for each residue.
        The psi angle is measured in the psi frame coordinate system.

        Parameters:
        -----------
        protein : Protein
            Input protein structure

        Returns:
        --------
        np.ndarray
            Array of shape [n_residues, 2] containing [sin(psi), cos(psi)]
        """
        # Create backbone rigid frames from N, CA, C
        psi_atoms = protein.atom_positions[:, [2, 1, 0], :]  # N, CA, C
        bb_rigid = RigidBuilder.from_matrix_local_to_global(psi_atoms)

        # Default psi frame transformation (from DEFAULT_FRAMES frame 3)
        n_residues = protein.atom_positions.shape[0]
        default_rot = DefaultFrames.PSI_ROT
        default_trans = DefaultFrames.PSI_TRANS

        default_psi_rigid = Rigid(
            np.tile(default_trans, (n_residues, 1)),
            Rotation(np.tile(default_rot[None, :, :], (n_residues, 1, 1)))
        )

        # Combined transformation: backbone -> psi frame
        psi_frame = bb_rigid.compose(default_psi_rigid)

        # Transform oxygen into the psi frame coordinate system
        oxygen_atom_rel_pos = psi_frame.invert().apply(protein.atom_positions[:, 3, :])

        # Extract y,z coordinates in psi frame (rotation is around x-axis)
        oxygen_atom_z_y = np.stack([oxygen_atom_rel_pos[:, 2], oxygen_atom_rel_pos[:, 1]], axis=-1)

        denom = np.sqrt(
            np.sum(
                np.square(oxygen_atom_z_y),
                axis=-1,
                keepdims=True
            )
            + 1e-8
        )
        psi_sin_cos = oxygen_atom_z_y / denom

        return psi_sin_cos
