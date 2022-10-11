#!/usr/bin/env python

import sys
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from object_detector_server.srv import GetObjPose
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path



if __name__ == "__main__":
    rospy.wait_for_service('obj_pose_estimator')
    try:
        get_pose = rospy.ServiceProxy('obj_pose_estimator', GetObjPose)
        pose_msg = get_pose()
        rospy.loginfo("Object Pose Received")
        print(pose_msg)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)