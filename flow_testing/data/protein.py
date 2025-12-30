from typing import Literal
import biotite.structure.io as bs_io
from flow_testing.data.protein_constants import PDB_CHAIN_IDS, restype_1to3, restype_name_to_atom14_names, restype_3to1, restypes
from biotite.structure import AtomArray, residue_iter
from dataclasses import dataclass
import numpy as np

from flow_testing.data.rigid import Rigid, matrix_to_rigids

@dataclass
class Protein:
    """
    Class that contains all the relevant information about a protein.
    Based on the atom_14 representation.
    """
    # shape : [n, 14, 3]
    aa_type: np.ndarray
    atom_positions: np.ndarray
    atom_names  : np.ndarray
    atom_mask : np.ndarray
    chain_ids: np.ndarray
    residue_ids: np.ndarray
    b_factors: np.ndarray

    @classmethod
    def from_pdb(cls, pdb_path: str):
        """
        Load a protein from a PDB file.
        """
        structure = bs_io.load_structure(pdb_path, extra_fields=['b_factor'])
        return cls.from_biotite(structure)

    @classmethod
    def from_backbone(cls, backbone: np.ndarray):
        """
        Create a protein from a backbone matrix of positions.
        Assumes each residue is an alanine.
        """
        num_residues = backbone.shape[0]

        atom_positions = np.zeros((num_residues, 14, 3))
        aa_type = np.zeros(num_residues, dtype=int)
        atom_names =  np.zeros((num_residues, 14), dtype=np.dtype('U6'))
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

        return cls(aa_type, atom_positions, atom_names, atom_mask, chain_ids, residue_ids, b_factors)

    @classmethod
    def from_biotite(cls, structure: AtomArray):
        """
        Load a protein from a Biotite AtomArray.
        """
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
    
        return cls(aa_type, atom_positions, atom_names, atom_mask, chain_ids, residue_ids, b_factors)

    def to_biotite(self) -> AtomArray:
        """
        Convert the protein to a Biotite AtomArray.
        """
        num_atms = self.atom_positions[self.atom_mask].shape[0]
        template = AtomArray(num_atms)

        template.coord = self.atom_positions[self.atom_mask]
        template.atom_name = self.atom_names[self.atom_mask]

        template.res_id = np.concatenate([[self.residue_ids[i]] * sum(self.atom_mask[i]) for i in range(len(self.residue_ids))])
        template.res_name = np.concatenate([[restype_1to3[restypes[aa]]] * sum(self.atom_mask[i]) for i, aa in enumerate(self.aa_type)])
        template.chain_id = np.concatenate([[PDB_CHAIN_IDS[self.chain_ids[i]]] * sum(self.atom_mask[i]) for i in range(len(self.chain_ids))])
        template.b_factor = np.concatenate([[self.b_factors[i]] * sum(self.atom_mask[i]) for i in range(len(self.b_factors))])

        return template

    def to_pdb(self, pdb_path: str):
        """
        Save the protein to a PDB file.
        """
        return bs_io.save_structure(pdb_path, self.to_biotite())

    def to_bb_rigid(self, center : bool = True) -> Rigid:
        """
        Convert the protein to a Rigid object using the backbone atoms.
        ['C', 'CA', 'N']
        """
        # 0 -> N, 1 -> CA, 2 -> C
        bb_atoms = self.atom_positions[:, [2, 1, 0], :]
        if center:
            bb_atoms = bb_atoms - np.mean(bb_atoms, axis=0)

        rigid = matrix_to_rigids(bb_atoms)

        return rigid

    def center(self, type: Literal['backbone', 'all'] = 'backbone'):
        """
        Center the protein.
        Args:
            type: The type of center to use.
                'backbone' : Center the backbone atoms.
                'all' : Center all the atoms.
        Returns:
            self
        """
        if type == 'backbone':
            ca_atoms = self.atom_positions[:, 1, :]
            self.atom_positions = self.atom_positions - np.mean(ca_atoms, axis=0)
            self.atom_positions[~self.atom_mask] = 0
        elif type == 'all':
            self.atom_positions = self.atom_positions - np.mean(self.atom_positions, axis=0)
        return self

    def to_psi_sin_cos(self) -> Rigid:
        """
        Calculate the sin and cos of the psi angle for each residue.
        """
        psi_atoms = self.atom_positions[:, [0, 1, 2], :]

        rigid = matrix_to_rigids(psi_atoms)

        oxygen_atom_rel_pos = rigid.invert().apply(self.atom_positions[:, 3, :])

        # extract out the y,z coordinates of the oxygen atom
        oxygen_atom_y_z = np.stack([oxygen_atom_rel_pos[:, 1], oxygen_atom_rel_pos[:, 2]], axis=-1)

        denom = np.sqrt(
            np.sum(
                np.square(oxygen_atom_y_z),
                axis=-1,
                keepdims=True
            )
            + 1e-8
        )
        psi_sin_cos = oxygen_atom_y_z / denom

        return psi_sin_cos
        
if __name__ == "__main__":
    from flow_testing.data.utils import calculate_backbone
    fp = '/home/luke/code/flow_testing/test-data/pdbs/8UVY.pdb'
    bio_structure = bs_io.load_structure(fp)
    bs_io.save_structure('test-data/pdbs/8UVY_biotite.pdb', bio_structure)

    protein = Protein.from_pdb(fp)
    print(protein)
    protein.center(type='backbone')
    protein.to_pdb('test-data/pdbs/8UVY_converted.pdb')
    rigid = protein.to_bb_rigid()
    print(rigid)
    psi_sin_cos = protein.to_psi_sin_cos()
    print(psi_sin_cos)
    backbone = calculate_backbone(rigid, psi_sin_cos)
    print(backbone)
    print(backbone.shape)

    bb_protein = Protein.from_backbone(backbone)
    bb_protein.to_pdb('test-data/pdbs/8UVY_backbone_converted.pdb')
    print("Done")
