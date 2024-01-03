import numpy as np


def random_rotate_pointcloud_horizontally(pointcloud, theta=None):
    rotated = False
    if pointcloud.shape[-1] != 3:
        pointcloud = pointcloud.T
        rotated = True

    if theta is None:
        theta = np.random.rand() * 2 * np.pi

    cosval = np.cos(theta)
    sinval = np.sin(theta)
    rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])

    rotated_pointcloud = np.dot(pointcloud, rotation_matrix)

    if rotated:
        rotated_pointcloud = rotated_pointcloud.T

    return rotated_pointcloud, theta


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotate(vertices, faces):
    """
    vertices: [numpoints, 3]
    """
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:, [1, 2, 0]].dot(M).dot(N).dot(K), faces[:, [1, 2, 0]]
    return v, f
