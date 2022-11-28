# object_detector

ROS (noetic) package for object detection.
___

### Dependencies
___
- [__object_pose_estimation__](https://github.com/IASRobolab/object_pose_estimation)
- [__camera_calibration__](https://github.com/IASRobolab/camera_calibration)
- [__object_detector_srv__](https://github.com/IASRobolab/object_detector_srv)
- [__o3d_ros_point_cloud_converter__](https://github.com/IASRobolab/o3d_ros_point_cloud_converter)

## Functionalities:
- registration of object model point cloud
- object point cloud acquisition
- multi-camera support

## Usage
___

- __object_pcd_srv__: ROS node implementing a service that provides detected object point cloud
- __object_pose_srv__: ROS node implementing a service that provides detected object 6D pose w.r.t. main camera frame

## License
___
Distributed under the ```GPLv3``` License. See [LICENSE](LICENSE) for more information.

## Authors
___
The package is provided by:

- [Fabio Amadio](https://github.com/fabio-amadio) [Mantainer]
