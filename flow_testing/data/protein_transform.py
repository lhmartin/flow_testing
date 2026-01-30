"""Coordinate transformation utilities for Protein structures."""

from typing import TYPE_CHECKING, Literal
import numpy as np

if TYPE_CHECKING:
    from flow_testing.data.protein import Protein


class ProteinTransform:
    """Coordinate transformation operations for proteins."""

    @staticmethod
    def center(protein: 'Protein', center_type: Literal['backbone', 'all'] = 'backbone') -> 'Protein':
        """
        Center a protein's coordinates.

        Parameters:
        -----------
        protein : Protein
            Protein to center (modified in place)
        center_type : Literal['backbone', 'all']
            'backbone' : Center using CA atoms
            'all' : Center using all atoms

        Returns:
        --------
        Protein
            The centered protein (same object, for method chaining)
        """
        if center_type == 'backbone':
            ca_atoms = protein.atom_positions[:, 1, :]
            protein.atom_positions = protein.atom_positions - np.mean(ca_atoms, axis=0)
            protein.atom_positions[~protein.atom_mask] = 0
        elif center_type == 'all':
            protein.atom_positions = protein.atom_positions - np.mean(protein.atom_positions, axis=0)
        return protein
