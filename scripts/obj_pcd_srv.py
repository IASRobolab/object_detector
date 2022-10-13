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
from object_detector.srv import GetObjPcd, GetObjPcdResponse
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from object_pose_estimation.pose_estimator import PoseEstimator
import open3d as o3d
import pdb
from ctypes import * # convert float to uint32


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convert_cloud_from_o3d_to_ros(open3d_cloud, frame_id="camera"):
    # Set "header"
    header = Header()
    header.stamp = rospy.get_rostime()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points.tolist()
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255).astype('uint32') # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]
        points = points.tolist()
        colors = colors.tolist()
        cloud_data = []
        for i in range(len(points)):
            cloud_data.append(points[i]+[colors[i]])

    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

def convert_cloud_from_ros_to_ord(ros_cloud):
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

    # Check empty
    open3d_cloud = o3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

        # combine
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

    return  open3d_cloud

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
        o3d.visualization.draw_geometries([obj_pcd])
        ros_cloud = convert_cloud_from_o3d_to_ros(obj_pcd, frame_id=self.camera_frame_id)

        conv_pcd = convert_cloud_from_ros_to_ord(ros_cloud)
        o3d.visualization.draw_geometries([conv_pcd])


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
    voxel_size = 0.001                   # Voxel size for downsamping
    filt_type = 'STATISTICAL'
    filt_params_dict = {'nb_neighbors': 100, 'std_ratio': 0.2}

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