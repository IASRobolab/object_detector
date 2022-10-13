#!/usr/bin/env python

import sys
import copy
import rospy
import rospkg
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from object_detector_srv.srv import GetObjPcd, GetObjPcdResponse
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from object_pose_estimation.pose_estimator import PoseEstimator
import open3d as o3d
import pdb
from ctypes import * # convert float to uint32
from o3d_ros_point_cloud_converter.conversion import convert_cloud_from_o3d_to_ros
from o3d_ros_point_cloud_converter.conversion import convert_cloud_from_ros_to_o3d

class ObjectPCDServer:
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


    def handle_pcd_request(self, req):
        rospy.loginfo("Incoming request")
        obj_pcd, _= self.estimator.get_yolact_pcd(filt_type = self.filt_type, filt_params_dict = self.filt_params_dict, flg_volume_int = True)
        transl = obj_pcd.get_center()

        # print("# POINTS:", np.asarray(obj_pcd.points).shape)
        # o3d.visualization.draw_geometries([obj_pcd])
        # o3d.io.write_point_cloud('driller_test.ply', obj_pcd)

        ros_cloud = convert_cloud_from_o3d_to_ros(obj_pcd, frame_id=self.camera_frame_id)

        # # Check back-and-forth conversion
        # conv_pcd = convert_cloud_from_ros_to_o3d(ros_cloud)
        # o3d.visualization.draw_geometries([conv_pcd])


        t = TransformStamped()
        t.header.stamp = ros_cloud.header.stamp
        t.header.seq = self.seq
        t.header.frame_id = self.camera_frame_id
        t.child_frame_id = self.object_frame_id
        t.transform.translation.x = transl[0]
        t.transform.translation.y = transl[1]
        t.transform.translation.z = transl[2]
        # broadcast only the camera frame - PCD center translation
        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = 0
        t.transform.rotation.w = 1
        self.br.sendTransform(t)

        rospy.loginfo("Object PCD Sent Correctly")

        self.seq += 1

        return GetObjPcdResponse(ros_cloud)

    

if __name__ == "__main__":
    rospy.init_node('obj_pcd_server')
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
    voxel_size = 0.0005                   # Voxel size for downsamping

    # filt_type = 'STATISTICAL'
    # filt_params_dict = {'nb_neighbors': 100, 'std_ratio': 0.2}
    filt_type = None
    filt_params_dict = None


    get_pcd_srv = ObjectPCDServer(cameras_dict = cameras_dict,
                                  obj_label = obj_label,
                                  obj_model_path = obj_model_path,
                                  yolact_weights = yolact_weights, 
                                  voxel_size = voxel_size,
                                  ext_cal_path = ext_cal_path,
                                  filt_type = filt_type, 
                                  filt_params_dict = filt_params_dict)

    s = rospy.Service('get_object_pcd', GetObjPcd, get_pcd_srv.handle_pcd_request)
    rospy.loginfo("Get Object PCD Service Ready")
    rospy.spin()