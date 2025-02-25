#!/usr/bin/env python3
"""
DPVO Combined ROS2 Publisher Node

This node subscribes to the outputs of DPVO and publishes:
  1. A TF transform for the current drone pose.
  2. The full trajectory as a MarkerArray (line strip).
  3. Camera frustum markers (one per pose) that mimic the C++ viewer.
  4. A point cloud (sensor_msgs/PointCloud2) built from DPVO point/color data.
  5. An image stream (sensor_msgs/Image) from the DPVO pipeline.

All data are published in the "map" frame.
"""

import os
from multiprocessing import Process, Queue
from pathlib import Path
import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

import threading

# ROS2 and TF2 imports
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField, Image
import struct
from cv_bridge import CvBridge
import tf_transformations as tft

SKIP = 0

# There are 8 camera model points and 10 line segments.
CAM_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, 1.5],
        [1.0, -1.0, 1.5],
        [1.0, 1.0, 1.5],
        [-1.0, 1.0, 1.5],
        [-0.5, 1.0, 1.5],
        [0.5, 1.0, 1.5],
        [0.0, 1.2, 1.5],
    ],
    dtype=np.float32,
)

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]],
    dtype=np.int32,
)

CAM_SCALE = 0.05


def transform_camera_points(translation, quaternion, scale=CAM_SCALE):
    """
    Transforms the fixed camera model points using the provided translation and quaternion.
    Args:
      translation (array-like): [x, y, z]
      quaternion (array-like): [qw, qx, qy, qz]
      scale (float): Scale factor for the camera model.
    Returns:
      np.ndarray: Transformed points, shape (8, 3)
    """
    # tf_transformations expects quaternion as [qx, qy, qz, qw]
    q = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    T = tft.quaternion_matrix(q)  # 4x4 homogeneous transformation matrix
    T[0:3, 3] = translation
    pts = CAM_POINTS * scale
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_hom = np.hstack((pts, ones))
    transformed = (T @ pts_hom.T).T[:, :3]
    return transformed


def create_frustum_marker(pose, marker_id, is_latest=False):
    """
    Creates a Marker (LINE_LIST) representing the camera frustum for one pose.
    Args:
      pose (tuple): (translation, quaternion) where translation is [x,y,z] and quaternion is [qw,qx,qy,qz]
      marker_id (int): Unique marker ID.
      is_latest (bool): If True, use a distinct color.
    Returns:
      Marker: A visualization_msgs/Marker message.
    """
    translation, quaternion = pose
    pts = transform_camera_points(translation, quaternion)

    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = Clock().now().to_msg()
    marker.ns = "camera_frustums"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.01

    # Color: red for the latest pose, blue for others.
    if is_latest:
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    else:
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    marker.color.a = 1.0

    # Build line segments from CAM_LINES.
    for line in CAM_LINES:
        pt1 = Point()
        pt1.x, pt1.y, pt1.z = pts[line[0]].tolist()
        pt2 = Point()
        pt2.x, pt2.y, pt2.z = pts[line[1]].tolist()
        marker.points.append(pt1)
        marker.points.append(pt2)
    return marker


def create_point_cloud_msg(points, colors, frame_id="map"):
    msg = PointCloud2()
    msg.header.stamp = Clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.is_dense = True
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16  # 12 bytes for x,y,z + 4 bytes for rgb
    msg.row_step = msg.point_step * points.shape[0]
    buffer = []
    for i in range(points.shape[0]):
        x, y, z = points[i]
        # Get r,g,b as unsigned ints (0-255)
        r, g, b = colors[i].astype(np.uint8)
        # Pack r, g, b into a single 32-bit int
        rgb_int = (r << 16) | (g << 8) | b
        # Convert the int to a float
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        buffer.append(struct.pack("ffff", x, y, z, rgb_float))
    msg.data = b"".join(buffer)
    return msg


class DPVOCombinedPublisher(Node):
    def __init__(self):
        super().__init__("dpvo_combined_publisher")
        # TF broadcaster for current pose.
        self.br = TransformBroadcaster(self)
        # Publisher for trajectory line-strip markers.
        self.trajectory_marker_pub = self.create_publisher(
            MarkerArray, "trajectory_markers", 10
        )
        # Publisher for camera frustum markers.
        self.frustum_marker_pub = self.create_publisher(
            MarkerArray, "camera_frustums", 10
        )
        # Publisher for point cloud.
        self.point_cloud_pub = self.create_publisher(PointCloud2, "point_cloud", 10)
        # Publisher for image stream.
        self.image_pub = self.create_publisher(Image, "image_stream", 10)
        # Timer to periodically publish all messages.
        self.create_timer(0.5, self.timer_callback)

        self.trajectory = None
        self.pc_points = None
        self.pc_colors = None
        self.image_data = None

        self.bridge = CvBridge()

    def update_trajectory(self, trajectory):
        """Update the trajectory history."""
        self.trajectory = trajectory

    def update_point_cloud(self, points, colors):
        """Update the point cloud data."""
        self.pc_points = points
        self.pc_colors = colors

    def update_image(self, image):
        """Update the image data."""
        self.image_data = image

    def timer_callback(self):
        # Publish current TF transform (from the last pose in trajectory).
        if self.trajectory is not None and self.trajectory.positions_xyz.shape[0] > 0:
            pos = self.trajectory.positions_xyz[-1]
            quat = self.trajectory.orientations_quat_wxyz[-1]
            # Reorder quaternion from [qw,qx,qy,qz] to [qx,qy,qz,qw] for TF.
            tf_quat = [quat[1], quat[2], quat[3], quat[0]]
            t = TransformStamped()
            t.header.stamp = Clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "drone"
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.x = float(tf_quat[0])
            t.transform.rotation.y = float(tf_quat[1])
            t.transform.rotation.z = float(tf_quat[2])
            t.transform.rotation.w = float(tf_quat[3])
            self.br.sendTransform(t)

        # Publish trajectory as a line strip.
        if self.trajectory is not None:
            traj_marker = Marker()
            traj_marker.header.frame_id = "map"
            traj_marker.header.stamp = Clock().now().to_msg()
            traj_marker.ns = "trajectory_path"
            traj_marker.id = 0
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD
            traj_marker.scale.x = 0.05
            traj_marker.color.a = 1.0
            traj_marker.color.r = 0.0
            traj_marker.color.g = 1.0
            traj_marker.color.b = 0.0
            for pos in self.trajectory.positions_xyz:
                pt = Point()
                pt.x = float(pos[0])
                pt.y = float(pos[1])
                pt.z = float(pos[2])
                traj_marker.points.append(pt)
            ma = MarkerArray()
            ma.markers.append(traj_marker)
            self.trajectory_marker_pub.publish(ma)

        # Publish camera frustum markers.
        if self.trajectory is not None:
            frustum_array = MarkerArray()
            num = self.trajectory.positions_xyz.shape[0]
            for i in range(num):
                pos = self.trajectory.positions_xyz[i]
                quat = self.trajectory.orientations_quat_wxyz[i]
                # Create a marker for this pose.
                # For each frustum, use a distinct ID and color the latest red.
                marker = create_frustum_marker(
                    (pos, quat), marker_id=i, is_latest=(i == num - 1)
                )
                frustum_array.markers.append(marker)
            self.frustum_marker_pub.publish(frustum_array)

        # Publish point cloud.
        if self.pc_points is not None and self.pc_colors is not None:
            pc_msg = create_point_cloud_msg(
                self.pc_points, self.pc_colors, frame_id="map"
            )
            self.point_cloud_pub.publish(pc_msg)

        # Publish image stream.
        if self.image_data is not None:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(self.image_data, encoding="bgr8")
                img_msg.header.stamp = Clock().now().to_msg()
                img_msg.header.frame_id = "map"
                self.image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Image publishing failed: {e}")


@torch.no_grad()
def run(
    cfg,
    network,
    imagedir,
    calib,
    stride=1,
    skip=0,
    viz=False,
    timeit=False,
    ros_pub_node=None,
):
    slam = None
    queue = Queue(maxsize=8)

    trajectory_positions = []  # Each element: [x, y, z]
    trajectory_orientations = []  # Each element: [qw, qx, qy, qz]
    trajectory_tstamps = []  # Timestamps

    if os.path.isdir(imagedir):
        reader = Process(
            target=image_stream, args=(queue, imagedir, calib, stride, skip)
        )
    else:
        reader = Process(
            target=video_stream, args=(queue, imagedir, calib, stride, skip)
        )
    reader.start()

    while True:
        t, image, intrinsics = queue.get()
        if t < 0:
            break

        # Convert image and intrinsics to torch tensors on GPU.
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

        # Extract current pose from slam.pg.poses_
        if slam.n > 0:
            current_pose_tensor = slam.pg.poses_[slam.n - 1]
            current_pose_array = (
                current_pose_tensor.detach().cpu().numpy()
            )  # [x,y,z, qx,qy,qz,qw]
            translation = current_pose_array[:3]
            # Reorder quaternion from [qx,qy,qz,qw] to [qw,qx,qy,qz] for trajectory.
            quat = np.array(
                [
                    current_pose_array[6],
                    current_pose_array[3],
                    current_pose_array[4],
                    current_pose_array[5],
                ]
            )
            trajectory_positions.append(translation)
            trajectory_orientations.append(quat)
            trajectory_tstamps.append(t)
        else:
            print(f"Warning: No valid pose returned at timestamp {t}")

        if trajectory_positions:
            current_traj = PoseTrajectory3D(
                positions_xyz=np.array(trajectory_positions),
                orientations_quat_wxyz=np.array(trajectory_orientations),
                timestamps=np.array(trajectory_tstamps),
            )
            if ros_pub_node is not None:
                ros_pub_node.update_trajectory(current_traj)
                # Update point cloud from current DPVO data.
                pc_points = slam.pg.points_.cpu().numpy()[: slam.m]
                pc_colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]
                ros_pub_node.update_point_cloud(pc_points, pc_colors)
                # Update image (convert tensor to CPU numpy image in BGR).
                img_np = image.permute(1, 2, 0).cpu().numpy()
                ros_pub_node.update_image(img_np)

        torch.cuda.empty_cache()

    reader.join()
    points = slam.pg.points_.cpu().numpy()[: slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]
    return slam.terminate(), (points, colors, (*intrinsics, H, W))


def show_image(image, t=0):
    img = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", img / 255.0)
    cv2.waitKey(t)


# --- Main Entry Point ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="dpvo.pth")
    parser.add_argument("--imagedir", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--name", type=str, help="name your run", default="result")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--timeit", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable ROS TF, trajectory, point cloud, and image publishing",
    )
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_colmap", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    print("Running with config...")
    print(cfg)

    if args.plot:
        rclpy.init(args=None)
        ros_pub_node = DPVOCombinedPublisher()
        # Run the ROS event loop in a separate thread.
        spin_thread = threading.Thread(
            target=rclpy.spin, args=(ros_pub_node,), daemon=True
        )
        spin_thread.start()
    else:
        ros_pub_node = None

    (poses, tstamps), (points, colors, calib) = run(
        cfg,
        args.network,
        args.imagedir,
        args.calib,
        args.stride,
        args.skip,
        args.viz,
        args.timeit,
        ros_pub_node=ros_pub_node,
    )

    # Build the final trajectory from DPVO results.
    trajectory = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=tstamps,
    )

    if args.save_ply:
        from dpvo.plot_utils import save_ply

        save_ply(args.name, points, colors)
    if args.save_colmap:
        from dpvo.plot_utils import save_output_for_COLMAP

        save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)
    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(
            f"saved_trajectories/{args.name}.txt", trajectory
        )

    if args.plot and ros_pub_node is not None:
        ros_pub_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()
