#!/usr/bin/env python

import sys
import copy
import rospy
import rospkg
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped
from object_detector_srv.srv import GetObjPose, GetObjPoseResponse
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from object_pose_estimation.pose_estimator import PoseEstimator
import open3d as o3d


class PoseEstimatorServer:
    def __init__(self, cameras_dict, obj_label, obj_model_path, yolact_weights, voxel_size, filt_type, filt_params_dict,
                 ext_cal_path, chess_size = (5, 4), chess_square_size = 40, calib_loops = 400, flg_cal_wait_key = False, 
                 object_frame_id = 'object', camera_frame_id = 'camera'):

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
        self.object_frame_id = object_frame_id
        self.camera_frame_id = camera_frame_id

        self.br = tf2_ros.TransformBroadcaster()


    def handle_pose_request(self, req):
        rospy.loginfo("Incoming request")
        T_icp, _, _ = self.estimator.locate_object(filt_type = self.filt_type, filt_params_dict = self.filt_params_dict)
        rot_mat = copy.deepcopy(T_icp[0:3,0:3])
        r = R.from_matrix(rot_mat)
        quat = r.as_quat()
        transl = T_icp[0:3,3]

        t = TransformStamped()
        t.header.stamp = rospy.get_rostime()
        t.header.seq = self.seq
        t.header.frame_id = self.camera_frame_id
        t.child_frame_id = self.object_frame_id
        t.transform.translation.x = transl[0]
        t.transform.translation.y = transl[1]
        t.transform.translation.z = transl[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.br.sendTransform(t)

        rospy.loginfo("Object Pose Sent Correctly")

        self.seq += 1

        return GetObjPoseResponse(t)

    

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

    ext_cal_path = rospack.get_path('object_detector')+'/config/cam1_H_camX.pkl'

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
    rospy.loginfo("Get Object Pose Estimator Service Ready")
    rospy.spin()