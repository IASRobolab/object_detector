#!/usr/bin/env python

import sys
import rospy
import tf
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from object_detector_srv.srv import GetObjPcd
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path



if __name__ == "__main__":
    rospy.init_node('obj_pcd_client')
    pub = rospy.Publisher('object_point_cloud', PointCloud2, queue_size=10)
    rospy.wait_for_service('get_object_pcd')
    try:
        get_obj_pcd = rospy.ServiceProxy('get_object_pcd', GetObjPcd)
        res = get_obj_pcd()
        rospy.loginfo("Object PCD Received")
        pub.publish(res.object_pcd)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
