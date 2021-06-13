#!/usr/bin/env python
import rospy
import math
from math import pi
import numpy as np
import tf2_ros
import random
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Quaternion
from kdl_kine.kdl_kine_solver import robot_kinematic
from kdl_kine.urdf import *



class iiwa14_kinematic(object):

    def __init__(self):
        ##TODO: Fill in the DH parameters based on the xacro file (cw3/iiwa_description/urdf/iiwa14.xacro)
        self.DH_params = np.array([[0.0, -pi/2, 0.2025, 0.0],
                                   [0.0, pi/2, 0.0, 0.0],
                                   [0.0, pi/2, 0.42, 0.0],
                                   [0.0, -pi/2, 0.0, 0.0],
                                   [0.0, -pi/2, 0.4, 0.0],
                                   [0.0, pi/2, 0.0, 0.0],
                                   [0.0, 0.0, 0.126, 0.0]])

        self.current_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.current_joint_vel = [0.2, 0.3, 0.8, 0.1, 0.6, 0.5, 0.7]

        self.joint_limit_min = [-170 * pi / 180, -120 * pi / 180, -170 * pi / 180, -120 * pi / 180, -170 * pi / 180,
                                -120 * pi / 180, -175 * pi / 180]
        self.joint_limit_max = [170 * pi / 180, 120 * pi / 180, 170 * pi / 180, 120 * pi / 180, 170 * pi / 180,
                                120 * pi / 180, 175 * pi / 180]

        ##The mass of each link.
        self.mass = [4, 4, 3, 2.7, 1.7, 1.8, 0.3]

        ##Moment on inertia of each link, defined at the centre of mass.
        ##Each row is (Ixx, Iyy, Izz) and Ixy = Ixz = Iyz = 0.
        self.Ixyz = np.array([[0.1, 0.09, 0.02],
                              [0.05, 0.018, 0.044],
                              [0.08, 0.075, 0.01],
                              [0.03, 0.01, 0.029],
                              [0.02, 0.018, 0.005],
                              [0.005, 0.0036, 0.0047],
                              [0.001, 0.001, 0.001]])

        ##gravity
        self.g = 9.8

        self.trans_cm = np.array([[0.0, -0.03, 0.12],
                                  [-0.0003, -0.059, 0.042],
                                  [0.0, 0.03, 0.13+0.2045],
                                  [0.0, 0.067, 0.034],
                                  [-0.0001, -0.021, 0.076+0.1845],
                                  [0.0, -0.0006, 0.0004],
                                  [0.0, 0.0, 0.02+0.081]])

        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback,
                                                queue_size=5)

        self.pose_broadcaster = tf2_ros.TransformBroadcaster()

    def joint_state_callback(self, msg):
        for i in range(0, 7):
            self.current_joint_position[i] = msg.position[i]

        current_pose = self.forward_kine(self.current_joint_position, 7)
        self.broadcast_pose(current_pose)

    def dh_matrix_standard(self, a, alpha, d, theta):
        A = np.zeros((4, 4))

        A[0, 0] = np.cos(theta)
        A[0, 1] = -np.sin(theta) * np.cos(alpha)
        A[0, 2] = np.sin(theta) * np.sin(alpha)
        A[0, 3] = a * np.cos(theta)

        A[1, 0] = np.sin(theta)
        A[1, 1] = np.cos(theta) * np.cos(alpha)
        A[1, 2] = -np.cos(theta) * np.sin(alpha)
        A[1, 3] = a * np.sin(theta)

        A[2, 1] = np.sin(alpha)
        A[2, 2] = np.cos(alpha)
        A[2, 3] = d

        A[3, 3] = 1.0

        return A

    def rotmat2q(self, T):
        q = Quaternion()

        angle = np.arccos((T[0, 0] + T[1, 1] + T[2, 2] - 1) / 2)

        xr = T[2, 1] - T[1, 2]
        yr = T[0, 2] - T[2, 0]
        zr = T[1, 0] - T[0, 1]

        x = xr / np.sqrt(np.power(xr, 2) + np.power(yr, 2) + np.power(zr, 2))
        y = yr / np.sqrt(np.power(xr, 2) + np.power(yr, 2) + np.power(zr, 2))
        z = zr / np.sqrt(np.power(xr, 2) + np.power(yr, 2) + np.power(zr, 2))

        q.w = np.cos(angle / 2)
        q.x = x * np.sin(angle / 2)
        q.y = y * np.sin(angle / 2)
        q.z = z * np.sin(angle / 2)

        return q

    def broadcast_pose(self, pose):

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = 'iiwa_link_0'
        transform.child_frame_id = 'iiwa_ee'

        transform.transform.translation.x = pose[0, 3]
        transform.transform.translation.y = pose[1, 3]
        transform.transform.translation.z = pose[2, 3]
        transform.transform.rotation = self.rotmat2q(pose)

        self.pose_broadcaster.sendTransform(transform)

    ##Useful Transformation function
    def T_translation(self, t):
        T = np.identity(4)
        for i in range(0, 3):
            T[i, 3] = t[i]
        return T

    ##Useful Transformation function
    def T_rotationZ(self, theta):
        T = np.identity(4)
        T[0, 0] = np.cos(theta)
        T[0, 1] = -np.sin(theta)
        T[1, 0] = np.sin(theta)
        T[1, 1] = np.cos(theta)
        return T

    ##Useful Transformation function
    def T_rotationX(self, theta):
        T = np.identity(4)
        T[1, 1] = np.cos(theta)
        T[1, 2] = -np.sin(theta)
        T[2, 1] = np.sin(theta)
        T[2, 2] = np.cos(theta)
        return T

    ##Useful Transformation function
    def T_rotationY(self, theta):
        T = np.identity(4)
        T[0, 0] = np.cos(theta)
        T[0, 2] = np.sin(theta)
        T[2, 0] = -np.sin(theta)
        T[2, 2] = np.cos(theta)
        return T


    def forward_kine(self, joint, frame):
        T = np.identity(4)
        ##Add offset from the iiwa platform.S
        T[2, 3] = 0.1575

        ##TODO: Fill in this function to complete Q2.
        for i in range(0, frame):
            A = self.dh_matrix_standard(self.DH_params[i,0], self.DH_params[i,1], self.DH_params[i,2], joint[i]+self.DH_params[i,3])
            T = T.dot(A)

        return T


    def forward_kine_cm(self, joint, frame):
        ##TODO: Fill in this function to complete Q2.
        ## "joint" is a numpy array of double consisting of the joint value.
        ## "frame" is an integer indicating the frame you wish to calculate.
        ## The output is a numpy 4*4 matrix describing the transformation from the 'iiwa_link_0' frame to the centre of mass of the specified link.

        T = self.forward_kine(joint,frame-1)
        Rotzi_1 = self.T_rotationZ(joint[frame-1]+self.DH_params[frame-1][3])
        TransGi = self.T_translation(self.trans_cm[frame-1])
        T_cm = T.dot(Rotzi_1).dot(TransGi)
        return T_cm

    def get_jacobian(self, joint, frame):
        ##TODO: Fill in this function to complete Q2.
        ## "joint" is a numpy array of double consisting of the joint value.
        ## The output is a numpy 6*7 matrix describing the Jacobian matrix defining at each frame.
        # Initial the lower part of jacobian matrix

        J = np.zeros((6,7))
        Tp0z = self.forward_kine(joint, frame)
        p_l = Tp0z[:3,3]

        for i in range(frame):
            T_i = self.forward_kine(joint, i)
            z = T_i[:3, 2]
            p = T_i[:3, 3]
            J_p = np.cross(z, p_l - p)
            J[:3, i] = J_p
            J[3:, i] = z

        return J

    def get_jacobian_cm(self, joint, frame):
        ##TODO: Fill in this function to complete Q2.
        ## "joint" is a numpy array of double consisting of the joint value.
        ## "frame" is an integer indicating the frame you wish to calculate.
        ## The output is a numpy 6*7 matrix describing the Jacobian matrix defining at the centre of mass of the specified link.

        J_cm = np.zeros((6,7))
        Tp0z = self.forward_kine_cm(joint, frame)
        p_l = Tp0z[:3,3]

        for i in range(frame):
            T_i = self.forward_kine(joint, i)
            z = T_i[:3, 2]
            p = T_i[:3, 3]
            J_p_cm = np.cross(z, p_l - p)
            J_cm[:3, i] = J_p_cm
            J_cm[3:, i] = z

        return J_cm

    def rotmat2rotquat(self,R):
        rot_vector = np.zeros(3)
        # quat computation
        w = 0.5 * math.sqrt(np.trace(R)+1)
        x = (R[2,1] - R[1,2])/(4*w)
        y = (R[0,2] - R[2,0])/(4*w)
        z = (R[1,0] - R[0,1])/(4*w)
        
        quat = [x,y,z,w]
        # theta computation 
        theta = 2*math.acos(quat[3])
        if(theta == 0):
            return rotquat
        
        rotquat = quat[:3] / np.sin(theta/2)
        return rotquat

    def inverse_kine_ite(self, desired_pose, current_joint):
        ##TODO: Fill in this function to complete Q2.
        ## "desired_pose" is a numpy 4*4 matrix describing the transformation of the manipulator.
        ## "current_joint" is an array of double consisting of the joint value (works as a starting point for the optimisation).
        ## The output is numpy vector containing an optimised joint configuration for the desired pose.
        n = len(current_joint)
        
        rotmat = desired_pose[:3,:3]
        rotquat = self.rotmat2rotquat(rotmat)
        
        # form the desired position vector in 6x1 form
        x_desired = np.zeros(6)
        x_desired[0:3] = desired_pose[0:3,3]
        x_desired[3:6] = rotquat
        
        output = current_joint.copy()
        
        maxiter = 100
        limit = 1e-2
        alpha = 0.005
        i = 0
        
        while i < maxiter:
            crt_pose = self.forward_kine(current_joint, n)
            J = self.get_jacobian(current_joint,n)
            # rotation matrix extract and transform
            rotmat = crt_pose[:3,:3]
            rotquat = self.rotmat2rotquat(rotmat)
            # current x update
            crt_x = np.zeros(6)
            crt_x[0:3] = crt_pose[:3,3]
            crt_x[3:6] = rotquat
            # error update
            e = x_desired - crt_x
            J_pinv = np.linalg.pinv(J)
            output += alpha * J_pinv.dot(e)
            e = np.linalg.norm(e)
            
            i += 1
            # error threshold 
            if e < limit:
                break

        return output

    def inverse_kine_closed_form(self, desired_pose):
        ##TODO: Fill in this function to complete Q2.
        ## "desired_pose" is a numpy 4*4 matrix describing the transformation of the manipulator.
        ## The output is a numpy matrix consisting of the joint value for the desired pose.
        ## You may need to re-structure the input of this function.
        raise NotImplementedError() #Remove this line, once implemented everything

    def getB(self, joint):
        ##TODO: Fill in this function to complete Q2.
        ## "joint" is a numpy array of double consisting of the joint value.
        ## The output is a numpy 7*7 matrix.

        B = np.zeros([7,7])
        Ixyz = np.zeros([3,3])
        for i in range(7):
            idx = i+1
            Jp = self.get_jacobian_cm(joint,idx)[:3,:]
            Jo = self.get_jacobian_cm(joint,idx)[3:,:]
            O_R_Gi = self.forward_kine_cm(joint,idx)[:3,:3]
            J_li = O_R_Gi.dot(np.diag(self.Ixyz[i])).dot(O_R_Gi.T)
            result = self.mass[i]*(Jp.T).dot(Jp) + (Jo.T).dot(J_li).dot(Jo)
            B += result
        return B

    def getC(self, joint, vel):
        ##TODO: Fill in this function to complete Q2.
        ## "joint" is a numpy array of double consisting of the joint value.
        ## "vel" is a numpy array of double consisting of the joint velocity.
        ## The output is a numpy 7*7 matrix.
        e = 1e-5
        Bijk = []
        C = np.zeros((7,7))
        
        for i in range(7):
            B = self.getB(joint)
            joint[i] += e
            B_diff = self.getB(joint)
            B_ele = (B_diff - B) / e
            Bijk.append(B_ele)
        
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    bjk_qi = Bijk[i][j,k]
                    bij_qk = Bijk[k][i,j]

                    C[i,j] += (bij_qk - bjk_qi / 2) * vel[k]
        
        return C


    def getG(self, joint):
        ##TODO: Fill in this function to complete Q2.
        ## "joint" is a numpy array of double consisting of the joint value.
        ## The output is a numpy array 7*1.

        g0_T = np.array([0,0,-self.g])
        G = np.zeros([7,1])
        
        for i in range(7):
            gi = 0
            for j in range(7):
                frame = j+1
                J_cm = self.get_jacobian_cm(joint, frame)
                Pli = J_cm[0:3,i]
                gi += (-self.mass[j]*g0_T.dot(Pli))
            G[i] = gi
        return G



# if __name__ == '__main__':
#     rospy.init_node('kuka_dynamics_node')
#     rate = rospy.Rate(10)
#     iiwa_kine = robot_kinematic('iiwa_link_0', 'iiwa_link_ee')
#     iiwa14 = iiwa14_kinematic()

#     q = PyKDL.JntArray(7)
#     qdot = PyKDL.JntArray(7)

#     while not rospy.is_shutdown():
#         # Jacobian check
#         print('Jacobian: ')
#         print(iiwa14.get_jacobian(iiwa14.current_joint_position,7))
#         print('KDL Jacobian: ')
#         print(iiwa_kine.get_jacobian(iiwa_kine.current_joint_position))

#         # Forward Kinematic check
#         print('Forward Kinematic: ')
#         print(iiwa14.forward_kine(iiwa14.current_joint_position,7))
#         print('KDL Forward Kinematic: ')
#         print(iiwa_kine.forward_kinematics(iiwa_kine.current_joint_position))

#         # Jacobian check
#         print('Jacobian_cm: ')
#         print(iiwa14.get_jacobian_cm(iiwa14.current_joint_position,7))
#         # print('KDL Jacobian_cm: ')
#         # print(iiwa_kine.get_jacobian(iiwa_kine.current_joint_position))

#         # Forward Kinematic CoM check
#         # print('Forward Kinematic: ')
#         # print(iiwa14.forward_kine_cm(iiwa14.current_joint_position,7))
#         # print('KDL Forward Kinematic: ')
#         # print(iiwa_kine.forward_kinematics(iiwa_kine.current_joint_position))

#         # B matrix check
#         print('B matrix: ')
#         print(iiwa14.getB(iiwa14.current_joint_position))

#         print('KDL B matrix: ')
#         B = iiwa_kine.getB(iiwa14.current_joint_position)
#         print(B)

#         Bmat = np.zeros((7, 7))
#         for i in range(7):
#             for j in range(7):
#                 Bmat[i, j] = B[i, j]
#         print(Bmat)

#         # C matrix check
#         print('C matrix: ')
#         print(iiwa14.getC(iiwa14.current_joint_position, iiwa14.current_joint_vel))
#         print('C(q, qdot): ')
#         print(iiwa_kine.getC(iiwa14.current_joint_position, iiwa14.current_joint_vel))

#         # G matrix check
#         print('G matrix: ')
#         print(iiwa14.getG(iiwa14.current_joint_position))
#         print('g(q): ')
#         print(iiwa_kine.getG(iiwa14.current_joint_position))
        
#         rate.sleep()


