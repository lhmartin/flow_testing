"""I/O operations for Protein data structures."""

from typing import TYPE_CHECKING
import biotite.structure.io as bs_io
from biotite.structure import AtomArray, residue_iter
import numpy as np

from flow_testing.data.protein_constants import (
    PDB_CHAIN_IDS, restype_1to3, restype_name_to_atom14_names,
    restype_3to1, restypes
)

if TYPE_CHECKING:
    from flow_testing.data.protein import Protein


class ProteinIO:
    """Handles I/O operations for Protein objects."""

    @classmethod
    def from_pdb(cls, pdb_path: str) -> 'Protein':
        """
        Load a protein from a PDB file.

        Parameters:
        -----------
        pdb_path : str
            Path to the PDB file

        Returns:
        --------
        Protein
            Loaded protein structure
        """
        structure = bs_io.load_structure(pdb_path, extra_fields=['b_factor'])
        return cls.from_biotite(structure)

    @classmethod
    def from_biotite(cls, structure: AtomArray) -> 'Protein':
        """
        Load a protein from a Biotite AtomArray.

        Parameters:
        -----------
        structure : AtomArray
            Biotite atom array structure

        Returns:
        --------
        Protein
            Loaded protein structure
        """
        from flow_testing.data.protein import Protein

        # remove hetero atoms
        structure = structure[~structure.hetero]
        # Get the residues
        residues = [residue for residue in residue_iter(structure)]
        num_residues = len(residues)

        # setup template arrays
        atom_positions = np.zeros((num_residues, 14, 3))
        atom_names = np.zeros((num_residues, 14), dtype=np.dtype('U6'))
        atom_mask = np.zeros((num_residues, 14), dtype=bool)
        chain_ids = np.zeros((num_residues), dtype=int)
        residue_ids = np.zeros((num_residues), dtype=int)
        b_factors = np.zeros((num_residues))
        aa_type = np.zeros((num_residues), dtype=int)

        for i, residue in enumerate(residues):
            # Get the atom positions
            res_name = residue[0].res_name
            restype_idx = restypes.index(restype_3to1[res_name])
            atm_names = restype_name_to_atom14_names[res_name]
            chain_ids[i] = PDB_CHAIN_IDS.index(residue[0].chain_id)
            residue_ids[i] = residue[0].res_id
            b_factors[i] = residue[0].b_factor
            aa_type[i] = restype_idx
            for atom in residue:
                atm_pos = atm_names.index(atom.atom_name)
                atom_positions[i, atm_pos, :] = atom.coord
                atom_names[i, atm_pos] = atom.atom_name
                atom_mask[i, atm_pos] = True

        return Protein(aa_type, atom_positions, atom_names, atom_mask, chain_ids, residue_ids, b_factors)

    @staticmethod
    def from_backbone(backbone: np.ndarray) -> 'Protein':
        """
        Create a protein from a backbone matrix of positions.
        Assumes each residue is an alanine.

        Parameters:
        -----------
        backbone : np.ndarray
            Backbone positions of shape [n_residues, 5, 3]

        Returns:
        --------
        Protein
            Protein with backbone atoms set
        """
        from flow_testing.data.protein import Protein

        num_residues = backbone.shape[0]

        atom_positions = np.zeros((num_residues, 14, 3))
        aa_type = np.zeros(num_residues, dtype=int)
        atom_names = np.zeros((num_residues, 14), dtype=np.dtype('U6'))
        atom_mask = np.ones((num_residues, 14), dtype=bool)
        chain_ids = np.zeros((num_residues), dtype=int)
        residue_ids = np.arange(num_residues, dtype=int)
        b_factors = np.zeros((num_residues))

        for i in range(num_residues):
            atom_names[i, :] = restype_name_to_atom14_names['ALA']
            atom_mask[i, :] = atom_names[i, :] != ''

            chain_ids[i] = 0
            residue_ids[i] = i
            b_factors[i] = 0
            aa_type[i] = restypes.index('A')
            atom_positions[i, :, :][atom_mask[i, :]] = backbone[i, :, :]

            # swap O and Cb atoms
            atom_positions[i, 3, :], atom_positions[i, 4, :] = backbone[i, 4, :], backbone[i, 3, :]

        return Protein(aa_type, atom_positions, atom_names, atom_mask, chain_ids, residue_ids, b_factors)

    @staticmethod
    def to_biotite(protein: 'Protein') -> AtomArray:
        """
        Convert a protein to a Biotite AtomArray.

        Parameters:
        -----------
        protein : Protein
            Protein to convert

        Returns:
        --------
        AtomArray
            Biotite atom array representation
        """
        num_atms = protein.atom_positions[protein.atom_mask].shape[0]
        template = AtomArray(num_atms)

        template.coord = protein.atom_positions[protein.atom_mask]
        template.atom_name = protein.atom_names[protein.atom_mask]

        template.res_id = np.concatenate([[protein.residue_ids[i]] * sum(protein.atom_mask[i]) for i in range(len(protein.residue_ids))])
        template.res_name = np.concatenate([[restype_1to3[restypes[aa]]] * sum(protein.atom_mask[i]) for i, aa in enumerate(protein.aa_type)])
        template.chain_id = np.concatenate([[PDB_CHAIN_IDS[protein.chain_ids[i]]] * sum(protein.atom_mask[i]) for i in range(len(protein.chain_ids))])
        template.b_factor = np.concatenate([[protein.b_factors[i]] * sum(protein.atom_mask[i]) for i in range(len(protein.b_factors))])

        return template

    @staticmethod
    def to_pdb(protein: 'Protein', pdb_path: str) -> None:
        """
        Save a protein to a PDB file.

        Parameters:
        -----------
        protein : Protein
            Protein to save
        pdb_path : str
            Output file path
        """
        return bs_io.save_structure(pdb_path, ProteinIO.to_biotite(protein))
