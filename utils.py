from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np

def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix