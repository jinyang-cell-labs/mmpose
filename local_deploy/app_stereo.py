# Copyright (c) OpenMMLab. All rights reserved.
"""Live stereo body pose estimation (triangulated metric 3D) with Rerun.

    camera 0 ─┐
              ├─> person det + top-down 2D pose (per camera, shared models)
    camera 1 ─┘   -> triangulation (calibration from camera_params/)
                  -> metric 3D skeleton (COCO-17)
                  -> robot arm retargeting (retarget_coco.py)
                  -> Rerun viewer (2 camera views + 3D view)

Unlike app.py (monocular 2D->3D lifting, root-relative output), the 3D
skeleton here is metric and absolute in the calibrated world frame
(camera0, remapped to z-up for display). The skeleton keeps the COCO-17
keypoint format end to end; no H36M conversion is involved.
"""
import argparse
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample

from core.capture import StereoPair
from core.config import load_config, resolve_device
from core.pose2d import Pose2DFrontend, load_models
from core.triangulation import StereoTriangulator, load_stereo_rig
from core.triangulation import self_test as triangulation_self_test
from core.viz import (RobotViz, log_2d, log_3d, log_camera_frustums,
                      log_joint_angles)
from retarget import URDFRobot
from retarget_coco import CocoArmRetargeter
from retarget_coco import self_test as retarget_self_test

# The calibration world frame is camera0 (x right, y down, z forward).
# Remap it to the right-handed z-up frame the 3D view and retargeter use:
# x stays right, camera depth becomes +y, up becomes +z.
R_CAM0_TO_ZUP = np.array([
    [1., 0., 0.],
    [0., 0., 1.],
    [0., -1., 0.],
])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='local_deploy/config_stereo.yaml')
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run the pipeline self-tests and a few synthetic frames '
        'without cameras or a Rerun server, then exit.')
    return parser.parse_args()


def viz_transform(viz_cfg):
    """4x4 world (camera0 frame) -> display (z-up) transform."""
    yaw = np.deg2rad(float(viz_cfg.get('yaw_deg', 0.0)))
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    t = np.eye(4)
    t[:3, :3] = rz @ R_CAM0_TO_ZUP
    t[:3, 3] = viz_cfg.get('translation', [0.0, 0.0, 0.0])
    return t


def build_pipeline(cfg, device):
    """Returns (frontend0, frontend1, triangulator)."""
    det_cfg = cfg.setdefault('detection', {})
    if not det_cfg.get('single_person', True):
        print('Stereo triangulation matches people across views by the '
              'primary person only: forcing detection.single_person=true.')
    det_cfg['single_person'] = True

    detector, pose_estimator = load_models(cfg['models'], device)
    frontend0 = Pose2DFrontend(detector, pose_estimator, cfg)
    frontend1 = Pose2DFrontend(detector, pose_estimator, cfg)

    calib = cfg.get('calibration', {})
    cams = load_stereo_rig(
        calib.get('intrinsics', 'local_deploy/camera_params/intrinsics.yaml'),
        calib.get('extrinsics', 'local_deploy/camera_params/extrinsics.yaml'))
    print(f'Stereo rig: {cams[0].name} (world) + {cams[1].name}, '
          f'baseline {np.linalg.norm(cams[1].T_world_cam[:3, 3]):.3f} m')
    return frontend0, frontend1, StereoTriangulator(cams)


def build_robot(cfg):
    """Returns (RobotViz, CocoArmRetargeter) or (None, None) if disabled."""
    robot_cfg = cfg.get('robot', {})
    if not robot_cfg.get('enabled', False):
        return None, None
    robot = URDFRobot(robot_cfg.get('urdf', 'robot_model/robot.urdf'))
    retargeter = CocoArmRetargeter(
        robot,
        smoothing=robot_cfg.get('smoothing', 0.35),
        mirror=robot_cfg.get('mirror', False))
    print(f'Robot "{robot.name}": {len(robot.visuals)} visual links, '
          f'driving {2 * 4} arm joints')
    return RobotViz(robot, robot_cfg), retargeter


def primary_instance(pose_est_results):
    """The single tracked person of one view, or None."""
    for res in pose_est_results:
        if int(res.get('track_id', -1)) != -1:
            return res
    return None


def triangulate_pair(triangulator, res0, res1, tri_thr, t_viz):
    """Triangulate the primary person of both views into the display frame.

    Returns (kpts_viz (17, 3), valid (17,), scores (17,)) or None.
    """
    if res0 is None or res1 is None:
        return None
    k0 = np.asarray(res0.pred_instances.keypoints)[0]
    s0 = np.asarray(res0.pred_instances.keypoint_scores)[0]
    k1 = np.asarray(res1.pred_instances.keypoints)[0]
    s1 = np.asarray(res1.pred_instances.keypoint_scores)[0]
    points, valid, scores = triangulator.triangulate(k0, s0, k1, s1, tri_thr)
    if not valid.any():
        return None
    kpts_viz = points @ t_viz[:3, :3].T + t_viz[:3, 3]
    return kpts_viz, valid, scores


def make_pose3d_sample(kpts_viz, scores, track_id):
    """Wrap triangulated keypoints in a PoseDataSample so core.viz.log_3d
    can render them like the lifter output."""
    sample = PoseDataSample()
    inst = InstanceData()
    inst.keypoints = kpts_viz[None]
    inst.keypoint_scores = scores[None]
    sample.pred_instances = inst
    sample.set_field(int(track_id), 'track_id')
    return sample


def run_stream(pair, frontend0, frontend1, triangulator, cfg):
    skeleton = frontend0.skeleton  # COCO links, reused for the 3D skeleton
    kpt_thr = float(cfg.get('kpt_thr', 0.3))
    tri_thr = float(cfg.get('triangulation', {}).get('kpt_thr', 0.35))
    jpeg_quality = int(cfg.get('rerun', {}).get('jpeg_quality', 75))
    t_viz = viz_transform(cfg.get('viz', {}))

    rr.log('pose3d', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    robot_viz, retargeter = build_robot(cfg)

    frame_idx = 0
    t0 = time.monotonic()
    fps_t, fps_n = t0, 0
    frustums_logged = False
    while True:
        ok, f0, f1 = pair.read()
        if not ok:
            print('End of stream.')
            break
        if not frustums_logged:
            # adapt the intrinsics to the negotiated capture resolution
            # before anything is triangulated or drawn
            triangulator.cams[0].scale_to(f0.shape[1], f0.shape[0])
            triangulator.cams[1].scale_to(f1.shape[1], f1.shape[0])
            log_camera_frustums(triangulator.cams, t_viz)
            frustums_logged = True

        frame_idx += 1
        rr.set_time_sequence('frame', frame_idx)
        rr.set_time_seconds('time', time.monotonic() - t0)

        pose0 = frontend0.step(f0)
        pose1 = frontend1.step(f1)
        log_2d(f0, pose0, skeleton, kpt_thr, jpeg_quality, root='video0')
        log_2d(f1, pose1, skeleton, kpt_thr, jpeg_quality, root='video1')

        res0, res1 = primary_instance(pose0), primary_instance(pose1)
        tri = triangulate_pair(triangulator, res0, res1, tri_thr, t_viz)
        if tri is not None:
            kpts_viz, valid, scores = tri
            sample = make_pose3d_sample(kpts_viz, scores,
                                        res0.get('track_id', 0))
            log_3d([sample], skeleton, kpt_thr)
            if retargeter is not None:
                angles = retargeter.update(kpts_viz, valid)
                if angles:
                    robot_viz.update(angles)
                    log_joint_angles(angles)
        else:
            log_3d([], skeleton, kpt_thr)

        fps_n += 1
        now = time.monotonic()
        if now - fps_t >= 5:
            n_valid = int(tri[1].sum()) if tri is not None else 0
            print(f'frame {frame_idx}: {fps_n / (now - fps_t):.1f} FPS, '
                  f'{len(pose0)}/{len(pose1)} person(s), '
                  f'{n_valid}/17 joints triangulated')
            fps_t, fps_n = now, 0


def smoke_test(cfg, device):
    """Exercise the full pipeline on synthetic frames, without cameras or a
    Rerun server, after running the geometry self-tests."""
    rr.init('mmpose_stereo_smoke_test')
    triangulation_self_test()
    retarget_self_test()
    frontend0, frontend1, triangulator = build_pipeline(cfg, device)
    skeleton = frontend0.skeleton
    t_viz = viz_transform(cfg.get('viz', {}))
    robot_viz, retargeter = build_robot(cfg)
    rng = np.random.default_rng(0)
    for idx in range(3):
        f0 = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        f1 = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        rr.set_time_sequence('frame', idx)
        pose0 = frontend0.step(f0)
        pose1 = frontend1.step(f1)
        log_2d(f0, pose0, skeleton, 0.3, 75, root='video0')
        log_2d(f1, pose1, skeleton, 0.3, 75, root='video1')
        tri = triangulate_pair(triangulator, primary_instance(pose0),
                               primary_instance(pose1), 0.35, t_viz)
        if tri is not None and retargeter is not None:
            angles = retargeter.update(tri[0], tri[1])
            if angles:
                robot_viz.update(angles)
        print(f'smoke test frame {idx}: '
              f'{len(pose0)}/{len(pose1)} detection(s)')
    print(f'smoke test OK (robot {"on" if robot_viz is not None else "off"})')


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'auto'))
    print(f'Using device: {device}')

    if args.smoke_test:
        smoke_test(cfg, device)
        return

    rerun_cfg = cfg.get('rerun', {})
    web_port = int(rerun_cfg.get('web_port', 9090))
    ws_port = int(rerun_cfg.get('ws_port', 9877))
    rr.init(rerun_cfg.get('app_id', 'mmpose_stereo'))
    robot_on = cfg.get('robot', {}).get('enabled', False)
    cam_views = rrb.Vertical(
        rrb.Spatial2DView(origin='video0', name='Camera 0'),
        rrb.Spatial2DView(origin='video1', name='Camera 1'))
    view_3d = rrb.Spatial3DView(
        origin='pose3d', name='Triangulated 3D pose + robot')
    if robot_on:
        right = rrb.Vertical(
            view_3d,
            rrb.TimeSeriesView(origin='angles', name='Joint angles'),
            row_shares=[3, 1])
    else:
        right = view_3d
    blueprint = rrb.Blueprint(
        rrb.Horizontal(cam_views, right, column_shares=[1, 2]),
        collapse_panels=True,
    )
    # rerun-sdk 0.18 names this rr.serve(); newer SDKs renamed it serve_web()
    rr.serve(
        open_browser=False,
        web_port=web_port,
        ws_port=ws_port,
        default_blueprint=blueprint,
        server_memory_limit=str(rerun_cfg.get('memory_limit', '1GB')))
    print(f'Rerun viewer: http://localhost:{web_port}?url=ws://localhost:{ws_port}')

    print('Loading models (checkpoints are downloaded on first run)...')
    frontend0, frontend1, triangulator = build_pipeline(cfg, device)
    cams_cfg = cfg.get('cameras', {})
    pair = StereoPair(cams_cfg.get('cam0', {}), cams_cfg.get('cam1', {}))
    print('Streaming. Press Ctrl+C to stop.')
    try:
        run_stream(pair, frontend0, frontend1, triangulator, cfg)
    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        pair.release()

    # Keep the Rerun server alive so the recording can still be inspected
    # after the streams have ended.
    print('Keeping Rerun server alive; press Ctrl+C to quit.')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
