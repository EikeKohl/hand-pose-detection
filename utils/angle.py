import numpy as np
import math

"""
This script contains helper functions to calculate angles between three points.
"""


def create_p1_p2_p3(p1_joint, p2_joint, p3_joint, annotations_output, idx=None):

    """
    This method takes the coordinate output of the mediapipe hand landmark prediction model and creates tuples for
    each joint as follows: (x,y).

    Parameters
    ----------
    p1_joint: Index of the joint to be used (int)
    p2_joint: Index of the joint to be used (int)
    p3_joint: Index of the joint to be used (int)
    annotations_output: Mediapipe hand landmark estimation output as a pd.DataFrame
    idx: Technical variable used to annotate data

    Returns
    -------
    Coordinates for the points p1, p2, and p3 to calculate an angle in find_angle().
    """

    if idx is not None:
        p1 = (
            annotations_output[f"{p1_joint}_x"][idx],
            annotations_output[f"{p1_joint}_y"][idx],
        )
        p2 = (
            annotations_output[f"{p2_joint}_x"][idx],
            annotations_output[f"{p2_joint}_y"][idx],
        )
        p3 = (
            annotations_output[f"{p3_joint}_x"][idx],
            annotations_output[f"{p3_joint}_y"][idx],
        )

    else:
        p1 = (annotations_output[f"{p1_joint}_x"], annotations_output[f"{p1_joint}_y"])
        p2 = (annotations_output[f"{p2_joint}_x"], annotations_output[f"{p2_joint}_y"])
        p3 = (annotations_output[f"{p3_joint}_x"], annotations_output[f"{p3_joint}_y"])

    return p1, p2, p3


def find_angle(p1, p2, p3):

    """
    This method calculates an angle between three points with coordinates. p2 is used as the vertex of the angle.

    Parameters
    ----------
    p1: End of the leg of the angle (tuple)
    p2: Vertex point of the angle (tuple)
    p3: End of the leg of the angle (tuple)

    Returns
    -------
    The calculated angle between p1, p2, and p3.
    """

    BAx = p1[0] - p2[0]
    BAy = p1[1] - p2[1]

    BCx = p3[0] - p2[0]
    BCy = p3[1] - p2[1]

    a = [BAx, BAy]
    b = [BCx, BCy]
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)

    theta = np.arccos(np.dot(a, b) / (a_mag * b_mag))

    return math.degrees(theta)
