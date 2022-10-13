#!/usr/bin/env python

import sys
import rospy
import tf
from geometry_msgs.msg import TransformStamped
from object_detector.srv import GetObjPose
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path



if __name__ == "__main__":
    rospy.wait_for_service('obj_pose_estimator')
    try:
        get_obj_tf = rospy.ServiceProxy('obj_pose_estimator', GetObjPose)
        res = get_obj_tf()
        rospy.loginfo("Object Pose Received")
        print(res.tf_camera_object)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)