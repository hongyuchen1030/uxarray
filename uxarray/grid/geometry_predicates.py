import numpy as np
from uxarray.grid.utils import cross_fma

def _is_pole_within_polygon(pole_string, polygon):
    """
    Determine if a Pole Point is Inside a Convex Oriented Face.

    Parameters:
    pole_string (str): "north_pole" or "south_pole".
    polygon (np.ndarray): Array of vertices in 3D cartesian coordinates for a polygon in counterclockwise order.

    Returns:
    int: -1 if pole point is outside the face, 0 if on an edge, 1 if inside the face.
    """
    pole = np.array([0, 0, 1])
    if pole_string == "north_pole":
        pole = np.array([0, 0, 1])
    elif pole_string == "south_pole":
        pole = np.array([0, 0, 1])
    else:
        raise ValueError("pole_string must be 'north_pole' or 'south_pole'.")

    # Make the vertices form a closed loop
    vertexes = np.vstack((polygon, polygon[0]))
    direction = 1.0

    for i in range(len(vertexes) - 1):
        vs = vertexes[i]  # Source vertex
        vt = vertexes[i + 1]  # Target vertex
        direction *= np.dot(np.cross(vs, vt), pole)

        if direction < 0.0:
            return -1  # Pole point is outside the face

        if direction == 0.0:
            return 0  # Pole point is on one of the edges of the face



