import biotite.structure.io as bs_io
from biotite.structure import AtomArray, residue_iter
from dataclasses import dataclass
import numpy as np

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
    
    psi_angles: np.ndarray | None = None

    @classmethod
    def from_pdb(cls, pdb_path: str):
        """
        Load a protein from a PDB file.
        """
        structure = bs_io.load_structure(pdb_path)
        return cls.from_biotite(structure)

    @classmethod
    def from_biotite(cls, structure: AtomArray):
        """
        Load a protein from a Biotite AtomArray.
        """
        _positions = structure.coord


if __name__ == "__main__":
    fp = '../test-data/pdbs/9VYX.pdb'
    protein = Protein.from_pdb(fp)
    print(protein)