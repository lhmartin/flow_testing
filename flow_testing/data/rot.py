import numpy as np

class Rotation:
    """
    Class that contains the rotation of a rigid protein.
    """

    def __init__(self, rot_matrix: np.ndarray):
        self.rot_matrix = rot_matrix

    def __call__(self):
        return self.rot_matrix