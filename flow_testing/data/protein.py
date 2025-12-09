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

    def to_psi_sin_cos(self, center : bool = True) -> Rigid:
        """
        Calculate the sin and cos of the psi angle for each residue.
        """
        psi_atoms = self.atom_positions[:, [1, 2, 3], :]
        if center:
            psi_atoms = psi_atoms - np.mean(psi_atoms, axis=0)

        rigid = matrix_to_rigids(psi_atoms)

        oxygen_atom_rel_pos = rigid.invert().apply(psi_atoms[:, 2, :])

        # extract out the y,z coordinates of the oxygen atom
        oxygen_atom_y_z = np.stack([oxygen_atom_rel_pos[:, 2], oxygen_atom_rel_pos[:, 1]], axis=-1)

        denom = np.sqrt(
            np.sum(
                np.square(oxygen_atom_y_z[:, 0]),
                axis=-1,
                keepdims=True
            )
            + 1e-8
        )
        psi_sin_cos = oxygen_atom_y_z / denom

        return psi_sin_cos

        
if __name__ == "__main__":
    from flow_testing.data.utils import calculate_backbone
    fp = 'test-data/pdbs/9VYX.pdb'
    bio_structure = bs_io.load_structure(fp)
    bs_io.save_structure('test-data/pdbs/9VYX_biotite.pdb', bio_structure)

    protein = Protein.from_pdb(fp)
    print(protein)
    protein.to_pdb('test-data/pdbs/9VYX_converted.pdb')
    rigid = protein.to_bb_rigid(center=False)
    print(rigid)
    psi_sin_cos = protein.to_psi_sin_cos(center=False)
    print(psi_sin_cos)
    backbone = calculate_backbone(rigid, psi_sin_cos)