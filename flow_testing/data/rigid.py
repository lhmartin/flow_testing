import numpy as np
from flow_testing.data.rot import Rotation


def matrix_to_rigids(matrix: np.ndarray) -> 'Rigid':

    """
    Convert a matrix of 3D positions to a Rigid object.

    Expected shape: [n, 3, 3]
    Where the first axis is the end of the v1 vector, 
    the second axis is the basis vector and the third axis is the end of the v2 vector.

    Returns:
        Rigid: A Rigid object.
    """
    basis_vectors = matrix[:, 1, :]
    v1s = matrix[:, 0, :] - basis_vectors
    v2s = matrix[:, 2, :] - basis_vectors

    rots = []

    for (v1, v2) in zip(v1s, v2s):
        e1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        u2 = v2 - e1 * (e1.T * v2)
        e2 = v2 / np.linalg.norm(u2, axis=-1, keepdims=True)
        e3 = np.cross(e1, e2, axis=-1)
        rots.append(np.stack([e1, e2, e3], axis=-1))

    rot = Rotation(np.stack(rots, axis=0))
    trans = basis_vectors

    return Rigid(trans, rot)

class Rigid:
    """
    Class that contains the 3D position and rotation of rigid protein.
    """

    def __init__(self, trans: np.ndarray, rot: Rotation):
        self.trans = trans
        self.rot = rot

    def __call__(self):
        return self.trans, self.rot

    def invert(self):
        inverted_rot = self.rot.invert()
        inverted_trans = -1 * inverted_rot.apply(self.trans)
        return Rigid(inverted_trans, inverted_rot)

    def apply(self, vector: np.ndarray):
        """
        Apply the rigid transformation to a vector.
        Expected shape: [..., 3]
        Returns:
            np.ndarray: The transformed vector.
            Shape: [..., 3]
        """
        return self.rot.apply(vector) + self.trans

if __name__ == "__main__":
    print('AAAAAAAAAA')