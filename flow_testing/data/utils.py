from re import I
import numpy as np
from flow_testing.data.rigid import Rigid
from flow_testing.data.protein_constants import idealized_AA_positions, restype_1to3, restypes

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
    for atom_type in range(3):
        idealized_positions[:, atom_type, :] = bb_rigid.apply(idealized_positions[:, atom_type, :])

    # need to place the O atom.
    


    # return the idealized positions
    return idealized_positions





    