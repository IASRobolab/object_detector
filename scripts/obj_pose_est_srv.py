#!/usr/bin/env python

import sys
import copy
import rospy
import rospkg
import tf
from geometry_msgs.msg import PoseStamped
from object_detector_server.srv import GetObjPose, GetObjPoseResponse
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from object_pose_estimation.pose_estimator import PoseEstimator
import open3d as o3d


class PoseEstimatorServer:
    def __init__(self, cameras_dict, obj_label, obj_model_path, yolact_weights, voxel_size, filt_type, filt_params_dict,
                 ext_cal_path, chess_size = (5, 4), chess_square_size = 40, calib_loops = 400, flg_cal_wait_key = False, 
                 frame_id = 'world'):

        self.estimator = PoseEstimator(cameras_dict = cameras_dict,
                                       obj_label = obj_label,
                                       obj_model_path = obj_model_path,
                                       yolact_weights = yolact_weights, 
                                       voxel_size = voxel_size,
                                       ext_cal_path = ext_cal_path,
                                       chess_size = chess_size,
                                       chess_square_size = chess_square_size,
                                       calib_loops = calib_loops,
                                       flg_cal_wait_key = flg_cal_wait_key)
        self.filt_type = filt_type
        self.filt_params_dict = filt_params_dict                   

        self.seq = 0
        self.frame_id = frame_id

    def handle_pose_request(self, req):
        rospy.loginfo("Incoming request")
        T_icp, _, _ = self.estimator.locate_object(filt_type = self.filt_type, filt_params_dict = self.filt_params_dict)
        rot_mat = copy.deepcopy(T_icp[0:3,0:3])
        r = R.from_matrix(rot_mat)
        quat = r.as_quat()
        transl = T_icp[0:3,3]

        pose_msg = PoseStamped()

        pose_msg.header.seq = self.seq
        pose_msg.header.stamp = rospy.get_rostime()
        pose_msg.header.frame_id = self.frame_id

        pose_msg.pose.position.x = transl[0]
        pose_msg.pose.position.y = transl[1]
        pose_msg.pose.position.z = transl[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        rospy.loginfo("Object Pose Sent Correctly")

        self.seq += 1
        return GetObjPoseResponse(pose_msg)

    

if __name__ == "__main__":
    rospy.init_node('obj_pose_estimator_server')
    rospack = rospkg.RosPack()

    # Select Yolact weights
    # yolact_weights = str(Path.home()) + "/Code/yolact/weights/yolact_plus_resnet50_54_800000.pth"  # Standard weights
    yolact_weights = str(Path.home()) + "/Code/yolact/weights/yolact_plus_resnet50_drill_74_750.pth" # fine-tuning for drill

    # Select PCD model
    # model_path = str(Path.home()) + "/Code/object_pose_estimation/test/models/mouse.ply"
    # model_path = str(Path.home()) + "/Code/object_pose_estimation/test/models/cup.ply"
    model_path = str(Path.home()) + "/Code/object_pose_estimation/test/models/drill.ply"

    ext_cal_path = rospack.get_path('object_detector_server')+'/config/cam1_H_camX.pkl'

    cameras_dict = {'049122251418': 'REALSENSE', '023322062736': 'REALSENSE'} 
    obj_label = 'drill'                  # Yolact label to find
    obj_model_path = model_path          # Path to the PCD model
    yolact_weights = yolact_weights      # Path to Yolact weights
    voxel_size = 0.005                   # Voxel size for downsamping
    filt_type = 'STATISTICAL'
    filt_params_dict = {'nb_neighbors': 50, 'std_ratio': 0.2}

    pose_est_srv = PoseEstimatorServer(cameras_dict = cameras_dict,
                                       obj_label = obj_label,
                                       obj_model_path = obj_model_path,
                                       yolact_weights = yolact_weights, 
                                       voxel_size = voxel_size,
                                       ext_cal_path = ext_cal_path,
                                       filt_type = filt_type, 
                                       filt_params_dict = filt_params_dict)

    s = rospy.Service('obj_pose_estimator', GetObjPose, pose_est_srv.handle_pose_request)
    print("Object Pose Estimator Service Ready")
    rospy.spin()