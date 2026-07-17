# Copyright (c) OpenMMLab. All rights reserved.
"""Rerun logging helpers shared by the mono and stereo apps."""
import cv2
import numpy as np
import rerun as rr

# Fallback if the lifter checkpoint carries no skeleton definition (H36M, 17 kpts)
H36M_SKELETON = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
                 (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                 (8, 14), (14, 15), (15, 16)]

# Distinct RGB colors, cycled by track id
TRACK_PALETTE = [
    (255, 96, 88), (66, 135, 245), (76, 187, 23), (255, 189, 46),
    (171, 71, 188), (38, 198, 218), (255, 112, 67), (156, 204, 101),
]


def _track_color(res):
    track_id = int(res.get('track_id', 0))
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


def log_2d(frame_bgr, pose_est_results, skeleton, kpt_thr, jpeg_quality,
           root='video'):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rr.log(f'{root}/image', rr.Image(rgb).compress(jpeg_quality=jpeg_quality))

    points, point_colors = [], []
    strips, strip_colors = [], []
    boxes = []
    for res in pose_est_results:
        if int(res.get('track_id', 0)) == -1:
            continue
        color = _track_color(res)
        inst = res.pred_instances
        if 'bboxes' in inst:
            boxes.append(np.asarray(inst.bboxes).reshape(-1, 4)[0])
        kpts = np.asarray(inst.keypoints)[0]
        scores = np.asarray(inst.keypoint_scores)[0]
        visible = scores >= kpt_thr
        points.append(kpts[visible])
        point_colors += [color] * int(visible.sum())
        for a, b in skeleton:
            if a < len(visible) and b < len(visible) and visible[a] \
                    and visible[b]:
                strips.append(np.stack([kpts[a], kpts[b]]))
                strip_colors.append(color)

    if points and sum(len(p) for p in points):
        rr.log(
            f'{root}/keypoints_2d',
            rr.Points2D(
                np.concatenate(points), colors=point_colors, radii=4))
    else:
        rr.log(f'{root}/keypoints_2d', rr.Clear(recursive=False))
    if strips:
        rr.log(f'{root}/skeleton_2d',
               rr.LineStrips2D(strips, colors=strip_colors, radii=2))
    else:
        rr.log(f'{root}/skeleton_2d', rr.Clear(recursive=False))
    if boxes:
        rr.log(
            f'{root}/bboxes',
            rr.Boxes2D(
                array=np.stack(boxes),
                array_format=rr.Box2DFormat.XYXY,
                colors=[(220, 220, 220)]))
    else:
        rr.log(f'{root}/bboxes', rr.Clear(recursive=False))


def log_3d(pose_results_3d, skeleton, kpt_thr, root='pose3d'):
    points, point_colors = [], []
    strips, strip_colors = [], []
    for res in pose_results_3d:
        if int(res.get('track_id', 0)) == -1:
            continue
        color = _track_color(res)
        inst = res.pred_instances
        for kpts, scores in zip(
                np.asarray(inst.keypoints), np.asarray(inst.keypoint_scores)):
            visible = scores >= kpt_thr
            points.append(kpts[visible])
            point_colors += [color] * int(visible.sum())
            for a, b in skeleton:
                if a < len(visible) and b < len(visible) and visible[a] \
                        and visible[b]:
                    strips.append(np.stack([kpts[a], kpts[b]]))
                    strip_colors.append(color)

    if points and sum(len(p) for p in points):
        rr.log(
            f'{root}/keypoints',
            rr.Points3D(
                np.concatenate(points), colors=point_colors, radii=0.02))
    else:
        rr.log(f'{root}/keypoints', rr.Clear(recursive=False))
    if strips:
        rr.log(f'{root}/skeleton',
               rr.LineStrips3D(strips, colors=strip_colors, radii=0.012))
    else:
        rr.log(f'{root}/skeleton', rr.Clear(recursive=False))


def log_joint_angles(angles):
    for name, q in angles.items():
        joint, side = name.rsplit('_', 1)
        rr.log(f'angles/{side}/{joint}', rr.Scalar(q))


def log_camera_frustums(cams, T_viz_world, root='pose3d'):
    """Show the calibrated cameras in the 3D view (static)."""
    for cam in cams:
        t = T_viz_world @ cam.T_world_cam
        rr.log(
            f'{root}/cameras/{cam.name}',
            rr.Transform3D(translation=t[:3, 3], mat3x3=t[:3, :3]),
            static=True)
        if cam.size is not None:
            rr.log(
                f'{root}/cameras/{cam.name}',
                rr.Pinhole(image_from_camera=cam.K, resolution=cam.size),
                static=True)


class RobotViz:
    """Drive the URDF robot in the Rerun 3D view via forward kinematics.

    Each link's visual STL is logged once (static); per frame only the link
    transforms are updated.
    """

    def __init__(self, robot, robot_cfg):
        self.robot = robot
        yaw = np.deg2rad(float(robot_cfg.get('yaw_deg', 0.0)))
        cy, sy = np.cos(yaw), np.sin(yaw)
        self.base = np.eye(4)
        self.base[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        self.base[:3, 3] = robot_cfg.get('offset', [0.0, -0.9, 0.95])

        for link, (mesh_path, t_vis) in robot.visuals.items():
            rr.log(
                f'pose3d/robot/{link}/mesh',
                rr.Asset3D(path=mesh_path),
                static=True)
            rr.log(
                f'pose3d/robot/{link}/mesh',
                rr.Transform3D(
                    translation=t_vis[:3, 3], mat3x3=t_vis[:3, :3]),
                static=True)
        self.update({})  # show the zero pose until the first person appears

    def update(self, joint_angles):
        poses = self.robot.fk(joint_angles)
        for link in self.robot.visuals:
            t = self.base @ poses[link]
            rr.log(
                f'pose3d/robot/{link}',
                rr.Transform3D(translation=t[:3, 3], mat3x3=t[:3, :3]))
