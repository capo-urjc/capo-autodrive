import numpy as np

def signed_angle(v1, v2):
    """
    Calculate the signed angle (in radians) from vector v1 to vector v2.

    Args:
        v1: Tuple or list (x1, y1) representing the first vector.
        v2: Tuple or list (x2, y2) representing the second vector.

    Returns:
        Signed angle in radians.
    """
    x1, y1 = v1
    x2, y2 = v2

    # Compute determinant (sin component) and dot product (cos component)
    det = x1 * y2 - y1 * x2  # determinant
    dot = x1 * x2 + y1 * y2  # dot product

    # Compute the signed angle using atan2
    angle = np.arctan2(det, dot)  # Returns angle in radians

    return angle  # Can convert to degrees with np.degrees(angle)

def world_to_local_frame(ref_position, ref_yaw, target_position):
    """
    Transforms a world coordinate point to the reference local coordinate frame.

    Args:
        ref_position: Tuple (cx, cy) representing the car's world position.
        ref_yaw: Car's yaw angle in degrees (rotation around Z-axis in CARLA).
        target_position: Tuple (tx, ty) representing the target world position.

    Returns:
        Transformed (x', y') in the car's reference frame.
    """
    cx, cy = ref_position
    tx, ty = target_position
    theta = np.radians(ref_yaw)  # Convert yaw to radians

    # Translate world coordinates to car-centered coordinates
    dx, dy = tx - cx, ty - cy

    # Rotate to align with the car's frame
    x_local =  np.cos(-theta) * dx - np.sin(-theta) * dy
    y_local =  np.sin(-theta) * dx + np.cos(-theta) * dy

    return x_local, y_local