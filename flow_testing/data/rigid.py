import numpy as np
from flow_testing.data.rot import Rotation


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

    def copy(self):
        return Rigid(self.trans.copy(), self.rot.copy())

    def apply(self, vector: np.ndarray):
        """
        Apply the rigid transformation to a vector.
        Expected shape: [..., 3]
        Returns:
            np.ndarray: The transformed vector.
            Shape: [..., 3]
        """
        return self.rot.apply(vector) + self.trans

    def compose(self, other: 'Rigid'):
        """
        Compose two rigid transformations.
        """
        new_rotation = self.rot.compose(other.rot)
        new_trans = self.rot.apply(other.trans) + self.trans

        return Rigid(new_trans, new_rotation)


class RigidBuilder:
    """Builder class for constructing Rigid objects from various representations."""

    @staticmethod
    def from_matrix_global_to_local(matrix: np.ndarray) -> Rigid:
        """
        Convert a matrix of 3D positions to a Rigid object.
        This goes from the global coordinate system to the local coordinate system of the rigid body.

        Parameters:
        -----------
        matrix : np.ndarray
            Expected shape: [n, 3, 3]
            Where the first axis is the end of the v1 vector,
            the second axis is the basis vector and the third axis is the end of the v2 vector.

        Returns:
        --------
        Rigid
            A Rigid object representing the transformation.
        """
        basis_vectors = matrix[:, 1, :]
        v1s = matrix[:, 0, :] - basis_vectors
        v2s = matrix[:, 2, :] - basis_vectors

        rots = []
        trans = []

        for i, (v1, v2) in enumerate(zip(v1s, v2s)):
            e1 = v1 / np.linalg.norm(v1)
            u2 = v2 - np.dot(e1, v2) * e1
            e2 = u2 / np.linalg.norm(u2)
            e3 = np.cross(e1, e2)
            rot_matrix = np.stack([e1, e2, e3], axis=0)
            rots.append(rot_matrix)

            trans.append(-(rot_matrix @ basis_vectors[i]))

        rot = Rotation(np.stack(rots, axis=0))
        trans = np.stack(trans, axis=0)

        return Rigid(trans, rot)

    @staticmethod
    def from_matrix_local_to_global(matrix: np.ndarray) -> Rigid:
        """
        Convert a matrix of 3D positions to a Rigid object.
        This goes from the local coordinate system of the rigid body to the global coordinate system.

        Parameters:
        -----------
        matrix : np.ndarray
            Expected shape: [n, 3, 3]
            Where the first axis is the end of the v1 vector,
            the second axis is the basis vector and the third axis is the end of the v2 vector.

        Returns:
        --------
        Rigid
            A Rigid object representing the transformation.
        """
        rigid = RigidBuilder.from_matrix_global_to_local(matrix)
        return rigid.invert()


# Backward compatibility aliases
def matrix_to_rigids_global_to_local(matrix: np.ndarray) -> Rigid:
    """Deprecated: Use RigidBuilder.from_matrix_global_to_local instead."""
    return RigidBuilder.from_matrix_global_to_local(matrix)


def matrix_to_rigids_local_to_global(matrix: np.ndarray) -> Rigid:
    """Deprecated: Use RigidBuilder.from_matrix_local_to_global instead."""
    return RigidBuilder.from_matrix_local_to_global(matrix)


if __name__ == "__main__":
    print('AAAAAAAAAA')
