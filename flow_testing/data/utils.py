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
    template[:, 1, 1] = psi_angles[:, 1]
    template[:, 1, 2] = -psi_angles[:, 0]
    template[:, 2, 1] = psi_angles[:, 0]
    template[:, 2, 2] = psi_angles[:, 1]

    return Rotation(template)

def calculate_backbone_2(bb_rigid : Rigid, psi_angles : np.ndarray):
    """
    Calculate the position on the 3D positions of the backbone atoms given the rigid frame and the psi angles.
    """
    n_residues = psi_angles.shape[0]
    ideal_amino_acid = idealized_AA_positions[restype_1to3[restypes[0]]]
    bb_atoms_ideal = np.stack([list(ideal_amino_acid[x][2]) for x in range(5)])

    # create a template array of the idealized positions
    backbone = np.zeros((n_residues, 5, 3))

    # apply the rigid frame to the idealized positions for first 4 atoms
    for atom_idx in range(4):
        ideal_pos = np.tile(bb_atoms_ideal[atom_idx], (n_residues, 1))
        backbone[:, atom_idx, :] = bb_rigid.invert().apply(ideal_pos)

    # Default frame 2 transformation for the psi group (from DEFAULT_FRAMES)
    default_rot = np.array([
        [-0.3594,  0.9332,  0.0000],
        [ 0.9332,  0.3594,  0.0000],
        [ 0.0000,  0.0000, -1.0000]
    ])
    default_trans = np.array([-0.5250, 1.3630, 0.0000])
    
    # Create default psi frame rigid (same for all residues)
    default_psi_rigid = Rigid(
        np.tile(default_trans, (n_residues, 1)),
        Rotation(np.tile(default_rot[None, :, :], (n_residues, 1, 1)))
    )
    
    # Create psi rotation from torsion angles
    psi_rotation = psi_angles_to_rotation(psi_angles)
    psi_rot_rigid = Rigid(np.zeros((n_residues, 3)), psi_rotation)
    
    # Compose: default_frame → psi_rotation
    psi_frame = default_psi_rigid.compose(psi_rot_rigid)
    
    # Compose with backbone frame
    combined_rigid = bb_rigid.invert().compose(psi_frame)
    
    # Apply to oxygen position
    o_ideal = np.tile(bb_atoms_ideal[4], (n_residues, 1))
    backbone[:, 4, :] = combined_rigid.apply(o_ideal)

    return backbone

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
    n_residues = psi_angles.shape[0]

    ideal_amino_acid = idealized_AA_positions[restype_1to3[restypes[0]]]
    bb_atoms_ideal = np.stack([list(ideal_amino_acid[x][2]) for x in range(5)])

    # create a template array of the idealized positions
    backbone = np.zeros((n_residues, 5, 3))

    # apply the rigid frame to the idealized positions
    for atom_idx in range(4):
        ideal_pos = np.tile(bb_atoms_ideal[atom_idx], (n_residues, 1))
        backbone[:, atom_idx, :] = bb_rigid.invert().apply(ideal_pos)

    psi_rotation = psi_angles_to_rotation(psi_angles)
    
    # repeat the translation vector for each residue
    o_ideal = np.tile(bb_atoms_ideal[4], (n_residues, 1))
    oxy_trans = np.array([1.526, 0, 0])[None, :].repeat(n_residues, axis=0)
    oxy_rigid = Rigid(oxy_trans, psi_rotation)

    combined_rigid = bb_rigid.invert().compose(oxy_rigid)
    o_atom_position = combined_rigid.apply(o_ideal)

    # Extract the idealized position of the O atom
    backbone[:, 4, :] = o_atom_position

    return backbone