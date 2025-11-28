import numpy as np
from data.rot import Rotation
class Rigid:
    """
    Class that contains the 3D position and rotation of rigid protein.
    """

    def __init__(self, trans: np.ndarray, rot: Rotation):
        self.trans = trans
        self.rot = rot

    def __call__(self):
        return self.pos, self.rot