# Copyright (c) OpenMMLab. All rights reserved.
"""Live body pose estimation (2D keypoints + lifted 3D pose) with Rerun.

The pipeline mirrors demo/body3d_pose_lifter_demo.py:

    camera frame -> mmdet person detector -> top-down 2D pose estimator
                 -> 2D-to-3D pose lifter -> Rerun viewer

The 2D keypoints are drawn as an overlay on the video stream and the lifted
3D skeleton is shown in a 3D view, both streamed to a Rerun web viewer
(open http://localhost:<web_port> in a browser).

For the two-camera pipeline (triangulation instead of lifting), see
app_stereo.py.
"""
import argparse
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from core.capture import open_capture
from core.config import load_config, resolve_device
from core.lifting import TemporalLifter
from core.pose2d import Pose2DFrontend, load_models
from core.viz import (H36M_SKELETON, RobotViz, log_2d, log_3d,
                      log_joint_angles)
from retarget import ArmRetargeter, URDFRobot
from retarget import self_test as retarget_self_test


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='local_deploy/config.yaml')
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run the full pipeline on a few synthetic frames without a '
        'camera or Rerun server, then exit. Used to validate the image.')
    return parser.parse_args()


def build_pipeline(cfg, device):
    """Returns (frontend, lifter); lifter is None when lifting is disabled."""
    detector, pose_estimator = load_models(cfg['models'], device)
    frontend = Pose2DFrontend(detector, pose_estimator, cfg)
    lifter = None
    if cfg.get('lifting', {}).get('enabled', True):
        lifter = TemporalLifter(cfg, frontend.dataset_name, device)
    return frontend, lifter


def build_robot(cfg, lifting_enabled):
    """Returns (RobotViz, ArmRetargeter) or (None, None) if disabled."""
    robot_cfg = cfg.get('robot', {})
    if not robot_cfg.get('enabled', False):
        return None, None
    if not lifting_enabled:
        print('robot.enabled is set but lifting is disabled: '
              'no 3D skeleton to retarget from, robot viz is off.')
        return None, None
    robot = URDFRobot(robot_cfg.get('urdf', 'robot_model/robot.urdf'))
    retargeter = ArmRetargeter(
        robot,
        smoothing=robot_cfg.get('smoothing', 0.35),
        mirror=robot_cfg.get('mirror', False))
    print(f'Robot "{robot.name}": {len(robot.visuals)} visual links, '
          f'driving {2 * 4} arm joints')
    return RobotViz(robot, robot_cfg), retargeter


def log_robot(robot_viz, retargeter, pose_lift_results):
    if not pose_lift_results:
        return
    kpts = np.asarray(pose_lift_results[0].pred_instances.keypoints)[0]
    angles = retargeter.update(kpts)
    if not angles:
        return
    robot_viz.update(angles)
    log_joint_angles(angles)


def run_stream(cap, frontend, lifter, cfg):
    skeleton_2d = frontend.skeleton
    kpt_thr = float(cfg.get('kpt_thr', 0.3))
    jpeg_quality = int(cfg.get('rerun', {}).get('jpeg_quality', 75))

    skeleton_3d = None
    if lifter is not None:
        skeleton_3d = lifter.dataset_meta.get(
            'skeleton_links') or H36M_SKELETON
        rr.log('pose3d', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    robot_viz, retargeter = build_robot(cfg, lifter is not None)

    frame_idx = 0
    t0 = time.monotonic()
    fps_t, fps_n = t0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print('End of stream.')
            break
        frame_idx += 1
        rr.set_time_sequence('frame', frame_idx)
        rr.set_time_seconds('time', time.monotonic() - t0)

        pose2d = frontend.step(frame)
        pose3d = lifter.step(pose2d, frame.shape[:2]) if lifter else []
        log_2d(frame, pose2d, skeleton_2d, kpt_thr, jpeg_quality)
        if lifter is not None:
            log_3d(pose3d, skeleton_3d, kpt_thr)
        if robot_viz is not None:
            log_robot(robot_viz, retargeter, pose3d)

        fps_n += 1
        now = time.monotonic()
        if now - fps_t >= 5:
            print(f'frame {frame_idx}: {fps_n / (now - fps_t):.1f} FPS, '
                  f'{len(pose2d)} person(s)')
            fps_t, fps_n = now, 0


def smoke_test(cfg, device):
    """Exercise the full pipeline on synthetic frames, without camera/server."""
    rr.init('mmpose_body3d_smoke_test')
    frontend, lifter = build_pipeline(cfg, device)
    skeleton_2d = frontend.skeleton
    skeleton_3d = None
    if lifter is not None:
        skeleton_3d = lifter.dataset_meta.get(
            'skeleton_links') or H36M_SKELETON
    robot_viz, retargeter = build_robot(cfg, lifter is not None)
    if robot_viz is not None:
        retarget_self_test()
    rng = np.random.default_rng(0)
    for idx in range(3):
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        rr.set_time_sequence('frame', idx)
        pose2d = frontend.step(frame)
        pose3d = lifter.step(pose2d, frame.shape[:2]) if lifter else []
        log_2d(frame, pose2d, skeleton_2d, 0.3, 75)
        if lifter is not None:
            log_3d(pose3d, skeleton_3d, 0.3)
        if robot_viz is not None:
            log_robot(robot_viz, retargeter, pose3d)
        print(f'smoke test frame {idx}: {len(pose2d)} detection(s)')
    print(f'smoke test OK (lifting {"on" if lifter is not None else "off"}, '
          f'robot {"on" if robot_viz is not None else "off"})')


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
    rr.init(rerun_cfg.get('app_id', 'mmpose_body3d'))
    lifting_on = cfg.get('lifting', {}).get('enabled', True)
    robot_on = lifting_on and cfg.get('robot', {}).get('enabled', False)
    views = [rrb.Spatial2DView(origin='video', name='Camera + 2D keypoints')]
    if lifting_on:
        view_3d = rrb.Spatial3DView(
            origin='pose3d', name='Lifted 3D pose + robot')
        if robot_on:
            views.append(
                rrb.Vertical(
                    view_3d,
                    rrb.TimeSeriesView(origin='angles', name='Joint angles'),
                    row_shares=[3, 1]))
        else:
            views.append(view_3d)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(*views),
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
    frontend, lifter = build_pipeline(cfg, device)
    cap = open_capture(cfg.get('camera', {}))
    print('Streaming. Press Ctrl+C to stop.')
    try:
        run_stream(cap, frontend, lifter, cfg)
    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        cap.release()

    # Keep the Rerun server alive so the recording can still be inspected
    # after a video file has finished playing.
    print('Keeping Rerun server alive; press Ctrl+C to quit.')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
