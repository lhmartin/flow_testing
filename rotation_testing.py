import numpy as np

def rotate_vectors(vectors, rotation_matrices):
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
    
    if rotation_matrices.ndim != 3:
        raise ValueError(f"rotation_matrices must be 3D array, got shape {rotation_matrices.shape}")
    
    N, D = vectors.shape
    N_rot, D_rot1, D_rot2 = rotation_matrices.shape
    
    if N != N_rot:
        raise ValueError(f"Number of vectors ({N}) must match number of rotation matrices ({N_rot})")
    
    if D != D_rot1 or D != D_rot2:
        raise ValueError(f"Vector dimension ({D}) must match rotation matrix dimensions ({D_rot1}x{D_rot2})")
    
    # Method 1: Using einsum (most efficient)
    rotated_vectors = np.einsum('nij,nj->ni', rotation_matrices, vectors)
    
    return rotated_vectors


# Alternative implementation using batch matrix multiplication
def rotate_vectors_matmul(vectors, rotation_matrices):
    """
    Alternative implementation using matmul.
    Slightly less efficient but more explicit.
    """
    # Add extra dimension to vectors for batch matrix multiplication
    # Shape: (N, 3) -> (N, 3, 1)
    vectors_expanded = vectors[..., np.newaxis]
    
    # Batch matrix multiplication
    # (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
    rotated = rotation_matrices @ vectors_expanded
    
    # Remove the extra dimension
    # (N, 3, 1) -> (N, 3)
    return rotated.squeeze(-1)


# Example usage
if __name__ == "__main__":
    # Create 5 random 3D vectors
    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=float)
    
    # Create 5 rotation matrices (rotation around z-axis by different angles)
    angles = np.array([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2])
    rotation_matrices = np.zeros((5, 3, 3))
    
    for i, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrices[i] = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    # Rotate the vectors
    rotated = rotate_vectors(vectors, rotation_matrices)
    
    print("Original vectors:")
    print(vectors)
    print("\nRotation angles (degrees):")
    print(np.degrees(angles))
    print("\nRotated vectors:")
    print(rotated)
    
    # Verify both methods give same result
    rotated_alt = rotate_vectors_matmul(vectors, rotation_matrices)
    print("\nMethods match:", np.allclose(rotated, rotated_alt))
    print("Done")