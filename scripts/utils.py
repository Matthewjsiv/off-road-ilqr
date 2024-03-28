import numpy as np
import torch

# def transformed_lib(pose, trajs):
#
#     lib = trajs[:,:,:2].clone()
#
#     submat = pose[:3, :3]
#     yaw = -np.arctan2(submat[1, 0], submat[0, 0]) + np.pi/2
#
#     rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
#                                 [np.sin(yaw), np.cos(yaw)]])
#
#     points = lib.reshape(-1,2)
#
#     rotated_points = points @ rotation_matrix.T
#
#     points = rotated_points.reshape(trajs.shape[0],-1,2)
#
#     return points

def transformed_lib(pose, trajs):

    lib = trajs[:,:2].clone()

    submat = pose[:3, :3]
    yaw = -np.arctan2(submat[1, 0], submat[0, 0]) + np.pi/2

    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                [np.sin(yaw), np.cos(yaw)]])

    points = lib.reshape(-1,2)

    rotated_points = points @ rotation_matrix.T

    # points = rotated_points.reshape(trajs.shape[0],-1,2)

    return rotated_points

def pose_msg_to_se3(msg):
        # quaternion_msg = msg[3:7]
        #msg is in xyzw
        Q = np.array([msg[6], msg[3], msg[4], msg[5]])
        rot_mat = quaternion_rotation_matrix(Q)

        se3 = np.zeros((4, 4))
        se3[:3, :3] = rot_mat
        se3[0, 3] = msg[0]
        se3[1, 3] = msg[1]
        se3[2, 3] = msg[2]
        se3[3, 3] = 1

        return se3

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix. Copied from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
            This rotation matrix converts a point in the local reference
            frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])

    return rot_matrix
