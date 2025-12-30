import numpy as np


class Rotation:
    """
    Class that contains the rotation of a rigid protein.
    """

    def __init__(self, rot_mats: np.ndarray):
        self.rot_mats = rot_mats

    def __call__(self):
        return self.rot_mats
    
    def invert(self) -> 'Rotation':
        return Rotation(self.rot_mats.transpose(0, 2, 1))

    def apply(self, vectors: np.ndarray):
        """
        Rotate a batch of vectors by a batch of rotation matrices.
        
        Parameters:
        -----------
        vectors : np.ndarray
            Array of shape (N, 3) or (N, D) where N is the number of vectors
            and D is the dimensionality (typically 3 for 3D rotation)
        rotation_matrices : np.ndarray
            Array of shape (N, D, D) where N is the number of rotation matrices
            and D x D is the size of each rotation matrix
        
        Returns:
        --------
        rotated_vectors : np.ndarray
            Array of shape (N, D) containing the rotated vectors
        
        Examples:
        ---------
        >>> # Rotate 5 3D vectors by 5 rotation matrices
        >>> vectors = np.random.randn(5, 3)
        >>> rotation_matrices = np.random.randn(5, 3, 3)
        >>> rotated = rotate_vectors(vectors, rotation_matrices)
        >>> rotated.shape
        (5, 3)
        """
        # Validate inputs
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D array, got shape {vectors.shape}")
        
        if self.rot_mats.ndim != 3:
            raise ValueError(f"rotation_matrices must be 3D array, got shape {self.rot_mats.shape}")
        
        N, D = vectors.shape
        N_rot, D_rot1, D_rot2 = self.rot_mats.shape
        
        if N != N_rot:
            raise ValueError(f"Number of vectors ({N}) must match number of rotation matrices ({N_rot})")
        
        if D != D_rot1 or D != D_rot2:
            raise ValueError(f"Vector dimension ({D}) must match rotation matrix dimensions ({D_rot1}x{D_rot2})")
        
        rotated_vectors = np.einsum('nij,nj->ni', self.rot_mats, vectors)
        
        return rotated_vectors

    def compose(self, other: 'Rotation'):
        """
        Compose two rotations in to a new rotation.
        Expected shape: [n, 3, 3]
        NOTE: Order of composition is self * other.
        Args:
            other: The other rotation to compose with.
            Expected shape: [n, 3, 3]
        Returns:
            Rotation: The composed rotation.
            Expected shape: [n, 3, 3]
        """
        new_rotation = np.einsum('njk,nkl->njl', self.rot_mats, other.rot_mats)
        return Rotation(new_rotation)