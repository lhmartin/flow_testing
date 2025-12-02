import numpy as np


class Rotation:
    """
    Class that contains the rotation of a rigid protein.
    """

    def __init__(self, rot_matrix: np.ndarray):
        self.rot_matrix = rot_matrix

    def __call__(self):
        return self.rot_matrix
    
    def invert(self) -> 'Rotation':
        return Rotation(self.rot_matrix.transpose(0, 2, 1))

    def apply(self, vector: np.ndarray):
        """
        Apply the rotation to a vector.
        Expected shape: [..., 3]
        Returns:
            np.ndarray: The rotated vector.
            Shape: [..., 3]
        """
        assert vector.shape[-1] == 3, "Vector must be a 3D vector"
        # assert same first dimension as the vector
        assert vector.shape[0] == self.rot_matrix.shape[0], "First dimension of vector and rotation matrix must match"
        return np.einsum('ijk,ik->ij', self.rot_matrix, vector)