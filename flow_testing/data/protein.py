"""Protein data structure with backward-compatible method delegation."""

from typing import Literal
from dataclasses import dataclass
import numpy as np

from flow_testing.data.rigid import Rigid


@dataclass
class Protein:
    """
    Class that contains all the relevant information about a protein.
    Based on the atom_14 representation.

    This is a pure data class. For I/O operations, use ProteinIO.
    For frame calculations, use ProteinFrameBuilder.
    For coordinate transforms, use ProteinTransform.
    """
    # shape : [n, 14, 3]
    aa_type: np.ndarray
    atom_positions: np.ndarray
    atom_names: np.ndarray
    atom_mask: np.ndarray
    chain_ids: np.ndarray
    residue_ids: np.ndarray
    b_factors: np.ndarray

    # === Backward compatibility methods (delegate to new classes) ===

    @classmethod
    def from_pdb(cls, pdb_path: str) -> 'Protein':
        """Load a protein from a PDB file. Delegates to ProteinIO."""
        from flow_testing.data.protein_io import ProteinIO
        return ProteinIO.from_pdb(pdb_path)

    @classmethod
    def from_backbone(cls, backbone: np.ndarray) -> 'Protein':
        """Create a protein from backbone positions. Delegates to ProteinIO."""
        from flow_testing.data.protein_io import ProteinIO
        return ProteinIO.from_backbone(backbone)

    @classmethod
    def from_biotite(cls, structure) -> 'Protein':
        """Load from Biotite AtomArray. Delegates to ProteinIO."""
        from flow_testing.data.protein_io import ProteinIO
        return ProteinIO.from_biotite(structure)

    def to_biotite(self):
        """Convert to Biotite AtomArray. Delegates to ProteinIO."""
        from flow_testing.data.protein_io import ProteinIO
        return ProteinIO.to_biotite(self)

    def to_pdb(self, pdb_path: str):
        """Save to PDB file. Delegates to ProteinIO."""
        from flow_testing.data.protein_io import ProteinIO
        return ProteinIO.to_pdb(self, pdb_path)

    def to_bb_rigid(self, center: bool = True) -> Rigid:
        """Convert to backbone rigid frames. Delegates to ProteinFrameBuilder."""
        from flow_testing.data.protein_frames import ProteinFrameBuilder
        return ProteinFrameBuilder.to_bb_rigid(self, center)

    def to_psi_sin_cos(self) -> np.ndarray:
        """Calculate psi angles. Delegates to ProteinFrameBuilder."""
        from flow_testing.data.protein_frames import ProteinFrameBuilder
        return ProteinFrameBuilder.to_psi_sin_cos(self)

    def center(self, type: Literal['backbone', 'all'] = 'backbone') -> 'Protein':
        """Center the protein. Delegates to ProteinTransform."""
        from flow_testing.data.protein_transform import ProteinTransform
        return ProteinTransform.center(self, type)


if __name__ == "__main__":
    from flow_testing.data.utils import calculate_backbone
    import biotite.structure.io as bs_io

    fp = '/home/luke/code/flow_testing/test-data/pdbs/9VYX.pdb'
    bio_structure = bs_io.load_structure(fp)
    bs_io.save_structure('test-data/pdbs/9VYX_biotite.pdb', bio_structure)

    protein = Protein.from_pdb(fp)
    print(protein)
    protein.center(type='backbone')
    protein.to_pdb('test-data/pdbs/9VYX_converted.pdb')
    rigid = protein.to_bb_rigid()
    print(rigid)
    psi_sin_cos = protein.to_psi_sin_cos()
    print(psi_sin_cos)
    backbone = calculate_backbone(rigid, psi_sin_cos)
    print(backbone)
    print(backbone.shape)

    bb_protein = Protein.from_backbone(backbone)
    bb_protein.to_pdb('test-data/pdbs/9VYX_backbone_converted.pdb')
    print("Done")
