import numpy as np
import math

"""
This script contains helper functions to calculate angles between three points.
"""


def create_p1_p2_p3(p1_joint, p2_joint, p3_joint, annotations_output, idx=None):

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
