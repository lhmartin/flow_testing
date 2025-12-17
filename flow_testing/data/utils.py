from re import I
import numpy as np
from flow_testing.data.rigid import Rigid
from flow_testing.data.protein_constants import idealized_AA_positions, restype_1to3, restypes
from flow_testing.data.rot import Rotation

def psi_angles_to_rotation(psi_angles : np.ndarray):
    """
    Convert the psi angles to a rotation matrix.
    psi_angles is a 2D array of shape [n, 2]
    """
    template = np.zeros((psi_angles.shape[0], 3, 3))

    template[:, 0, 0] = 1
    template[:, 1, 1] = psi_angles[:, 0]
    template[:, 1, 2] = psi_angles[:, 1]
    template[:, 2, 1] = -psi_angles[:, 1]
    template[:, 2, 2] = psi_angles[:, 0]

    return Rotation(template)



def calculate_backbone(bb_rigid : Rigid, psi_angles : np.ndarray):
    """
    Calculate the position on the 3D positions of the backbone atoms given the rigid frame and the psi angles.

    Args:
        bb_rigid : Rigid
            The rigid frame of the backbone atoms.
            Shape: [n, 3, 3]
        psi_angles : np.ndarray
            The psi angles of the backbone atoms.
            Shape: [n, 1]

    Returns:
        np.ndarray: The 3D positions of the backbone atoms.
    """

    # apply the rigid frame to the idealized positions
    # set the AA type to Alanine as this is just an idealised backbone
    aas = np.zeros(psi_angles.shape[0], dtype=int)

    ideal_amino_acid = idealized_AA_positions[restype_1to3[restypes[0]]]
    bb_atoms = np.stack([list(ideal_amino_acid[x][2]) for x in range(4)])

    # create a template array of the idealized positions
    idealized_positions = np.array([bb_atoms for i in aas])

    # apply the rigid frame to the idealized positions
    for atom_type in range(4):
        idealized_positions[:, atom_type, :] = bb_rigid.apply(idealized_positions[:, atom_type, :])

    # need to place the O atom.
    
    oxygen_rotation = psi_angles_to_rotation(psi_angles)

    idealized_positions[:, 3, :] = oxygen_rotation.apply(idealized_positions[:, 3, :])

    # return the idealized positions
    return idealized_positions





    