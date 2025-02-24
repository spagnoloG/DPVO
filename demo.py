#!/usr/bin/env python3
import os
from multiprocessing import Process, Queue
from pathlib import Path
import threading
import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

# Import ROS2 modules once at the top.
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

SKIP = 0

########################################################################
# ROS2 Publisher Node for Continuous Trajectory Updates
########################################################################
class DPVORos2Publisher(Node):
    def __init__(self):
        super().__init__('dpvo_ros_publisher')
        self.publisher_ = self.create_publisher(Marker, 'trajectory_marker', 10)
        self.trajectory = None
        # Publish marker at a fixed rate (every 0.5 seconds)
        self.create_timer(0.5, self.timer_callback)

    def update_trajectory(self, trajectory):
        """Called from the DPVO loop to update the trajectory."""
        self.trajectory = trajectory

    def timer_callback(self):
        if self.trajectory is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width.
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Build marker points while skipping any None values.
        marker.points = []
        for pos in self.trajectory.positions_xyz:
            if pos is None:
                continue
            pt = Point()
            pt.x = float(pos[0])
            pt.y = float(pos[1])
            pt.z = float(pos[2])
            marker.points.append(pt)
        self.publisher_.publish(marker)
        self.get_logger().info("Published trajectory marker with {} points".format(len(marker.points)))


########################################################################
# DPVO Run Function with Continuous RViz Publishing
########################################################################
@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, ros_pub_node=None):
    slam = None
    queue = Queue(maxsize=8)

    trajectory_positions = [] 
    trajectory_tstamps = [] 

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    while True:
        (t, image, intrinsics) = queue.get()
        if t < 0:
            break

        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            # Process the frame.
            slam(t, image, intrinsics)

        # Extract the current pose from slam.pg.poses_.
        # In DPVO, poses are stored as 7-element vectors: [x, y, z, qx, qy, qz, qw].
        if slam.n > 0:
            current_pose_tensor = slam.pg.poses_[slam.n - 1]  # Get the latest pose.
            # Extract only the translation (first three components).
            current_pose = current_pose_tensor[:3].detach().cpu().numpy()
            trajectory_positions.append(current_pose)
            trajectory_tstamps.append(t)
        else:
            print("Warning: No valid pose returned at timestamp {}".format(t))

        # Build the current trajectory if at least one pose is available.
        if trajectory_positions:
            # Since DPVO does not seem to provide orientation for every frame,
            # we use dummy zeros here.
            dummy_orientations = np.zeros((len(trajectory_positions), 4))
            current_traj = PoseTrajectory3D(
                positions_xyz=np.array(trajectory_positions),
                orientations_quat_wxyz=dummy_orientations,
                timestamps=np.array(trajectory_tstamps)
            )
            if ros_pub_node is not None:
                ros_pub_node.update_trajectory(current_traj)

        torch.cuda.empty_cache()

    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


########################################################################
# Main Entry Point
########################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--save_colmap', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    # If --plot is enabled, start the ROS2 publisher in a separate thread.
    if args.plot:
        rclpy.init(args=None)
        ros_pub_node = DPVORos2Publisher()
        ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_pub_node,), daemon=True)
        ros_spin_thread.start()
    else:
        ros_pub_node = None

    (poses, tstamps), (points, colors, calib) = run(
        cfg, args.network, args.imagedir, args.calib,
        args.stride, args.skip, args.viz, args.timeit, ros_pub_node=ros_pub_node
    )
    
    # Build the final trajectory from DPVO results.
    trajectory = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=tstamps
    )

    if args.save_ply:
        save_ply(args.name, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)

    if args.plot:
        import time
        time.sleep(2)
        ros_pub_node.destroy_node()
        rclpy.shutdown()