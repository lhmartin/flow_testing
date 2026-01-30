"""Data structures for protein representation and manipulation."""

from flow_testing.data.protein import Protein
from flow_testing.data.protein_io import ProteinIO
from flow_testing.data.protein_frames import ProteinFrameBuilder
from flow_testing.data.protein_transform import ProteinTransform
from flow_testing.data.rigid import Rigid, RigidBuilder, matrix_to_rigids_global_to_local, matrix_to_rigids_local_to_global
from flow_testing.data.rot import Rotation
from flow_testing.data.utils import BackboneBuilder, calculate_backbone, psi_angles_to_rotation
from flow_testing.data.so3_utils import random_rotation_matrices, so3_geodesic_interpolation
from flow_testing.data.protein_constants import (
    PDB_CHAIN_IDS,
    restypes,
    restype_1to3,
    restype_3to1,
    restype_name_to_atom14_names,
    idealized_AA_positions,
    backbone_atoms,
    DefaultFrames,
)

__all__ = [
    # Core classes
    'Protein',
    'Rigid',
    'Rotation',
    # Builder/utility classes
    'ProteinIO',
    'ProteinFrameBuilder',
    'ProteinTransform',
    'RigidBuilder',
    'BackboneBuilder',
    # Utility functions
    'calculate_backbone',
    'psi_angles_to_rotation',
    'random_rotation_matrices',
    'so3_geodesic_interpolation',
    # Deprecated aliases (for backward compatibility)
    'matrix_to_rigids_global_to_local',
    'matrix_to_rigids_local_to_global',
    # Constants
    'PDB_CHAIN_IDS',
    'restypes',
    'restype_1to3',
    'restype_3to1',
    'restype_name_to_atom14_names',
    'idealized_AA_positions',
    'backbone_atoms',
    'DefaultFrames',
]
