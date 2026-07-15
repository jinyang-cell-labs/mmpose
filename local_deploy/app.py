# Copyright (c) OpenMMLab. All rights reserved.
"""Live body pose estimation (2D keypoints + lifted 3D pose) with Rerun.

The pipeline mirrors demo/body3d_pose_lifter_demo.py:

    camera frame -> mmdet person detector -> top-down 2D pose estimator
                 -> 2D-to-3D pose lifter -> Rerun viewer

The 2D keypoints are drawn as an overlay on the video stream and the lifted
3D skeleton is shown in a 3D view, both streamed to a Rerun web viewer
(open http://localhost:<web_port> in a browser).
"""
import argparse
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import yaml
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (_track_by_iou, _track_by_oks,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmpose.structures import PoseDataSample
from mmpose.utils import adapt_mmdet_pipeline

# Fallback if the lifter checkpoint carries no skeleton definition (H36M, 17 kpts)
H36M_SKELETON = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
                 (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                 (8, 14), (14, 15), (15, 16)]

# Distinct RGB colors, cycled by track id
TRACK_PALETTE = [
    (255, 96, 88), (66, 135, 245), (76, 187, 23), (255, 189, 46),
    (171, 71, 188), (38, 198, 218), (255, 112, 67), (156, 204, 101),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='local_deploy/config.yaml')
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run the full pipeline on a few synthetic frames without a '
        'camera or Rerun server, then exit. Used to validate the image.')
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_device(device):
    if device in (None, 'auto'):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def open_capture(cam_cfg):
    src = str(cam_cfg.get('path', '/dev/video0'))
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(
            f'Could not open video source {src!r}. If it is a camera, make '
            'sure the device is passed into the container (see bootstrap.sh) '
            'and not in use by another application.')
    # Force the pixel format BEFORE size/fps: many UVC cameras (e.g. Arducam
    # OV9782) only reach their full frame rate in MJPG, while OpenCV's V4L2
    # backend defaults to YUYV, which may be capped as low as 10 FPS.
    fourcc = str(cam_cfg.get('fourcc', 'MJPG'))
    if fourcc and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if cam_cfg.get('width'):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg['width']))
    if cam_cfg.get('height'):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg['height']))
    if cam_cfg.get('fps'):
        cap.set(cv2.CAP_PROP_FPS, float(cam_cfg['fps']))
    # keep at most one buffered frame so the stream stays live (low latency)
    # even when inference is slower than the camera frame rate
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f'Capture negotiated: {w:.0f}x{h:.0f} @ '
          f'{cap.get(cv2.CAP_PROP_FPS):.0f} FPS')
    return cap


class Body3DPipeline:
    """Person detector -> top-down 2D pose -> pose lifter, with tracking.

    Adapted from demo/body3d_pose_lifter_demo.py, but keeps only a bounded
    history of 2D results so a long-running webcam stream does not grow
    memory without limit.
    """

    def __init__(self, cfg, device):
        models = cfg['models']
        self.detector = init_detector(
            models['det_config'], models['det_checkpoint'], device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_estimator = init_model(
            models['pose2d_config'],
            models['pose2d_checkpoint'],
            device=device)

        lift_cfg = cfg.get('lifting', {})
        self.lifting_enabled = bool(lift_cfg.get('enabled', True))
        self.pose_lifter = None
        if self.lifting_enabled:
            self.pose_lifter = init_model(
                models['pose3d_config'],
                models['pose3d_checkpoint'],
                device=device)

        det_cfg = cfg.get('detection', {})
        self.det_cat_id = int(det_cfg.get('cat_id', 0))
        self.bbox_thr = float(det_cfg.get('bbox_thr', 0.5))
        self.single_person = bool(det_cfg.get('single_person', False))
        self._selected_bbox = None  # last primary-person bbox (single_person)

        track_cfg = cfg.get('tracking', {})
        self.tracking_thr = float(track_cfg.get('thr', 0.3))
        self._track = _track_by_oks if track_cfg.get('use_oks',
                                                     False) else _track_by_iou

        self.norm_pose_2d = bool(lift_cfg.get('norm_pose_2d', True))
        self.rebase_keypoint = bool(lift_cfg.get('rebase_keypoint', True))

        self.pose2d_dataset_name = self.pose_estimator.dataset_meta[
            'dataset_name']
        if self.lifting_enabled:
            lift_dataset = self.pose_lifter.cfg.test_dataloader.dataset
            self.seq_len = lift_dataset.get('seq_len', 1)
            self.seq_step = lift_dataset.get('seq_step', 1)
            self.causal = lift_dataset.get('causal', False)
            self.pose3d_dataset_name = self.pose_lifter.dataset_meta[
                'dataset_name']

        self._history = []  # converted 2D results of the last N frames
        self._results_last = []  # previous frame's 2D results, for tracking
        self._next_id = 0

    def _select_primary(self, bboxes):
        """Keep only the biggest person, with hysteresis toward the person
        selected in previous frames so the choice does not flip-flop between
        two similarly-sized people (which would corrupt the lifter's
        per-track 2D sequence)."""
        if len(bboxes) <= 1:
            if len(bboxes) == 1:
                self._selected_bbox = bboxes[0]
            return bboxes

        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best = int(np.argmax(areas))
        prev = self._selected_bbox
        if prev is not None:
            # IoU of each candidate with the previously selected person
            x1 = np.maximum(bboxes[:, 0], prev[0])
            y1 = np.maximum(bboxes[:, 1], prev[1])
            x2 = np.minimum(bboxes[:, 2], prev[2])
            y2 = np.minimum(bboxes[:, 3], prev[3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            prev_area = (prev[2] - prev[0]) * (prev[3] - prev[1])
            ious = inter / (areas + prev_area - inter + 1e-6)
            cand = int(np.argmax(ious))
            # stick with the tracked person unless someone is clearly bigger
            if ious[cand] > 0.3 and areas[cand] >= 0.7 * areas[best]:
                best = cand
        self._selected_bbox = bboxes[best]
        return bboxes[[best]]

    def step(self, frame_bgr):
        """Process one frame. Returns (pose_est_results, pose_lift_results)."""
        det_result = inference_detector(self.detector, frame_bgr)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(
            pred_instance.labels == self.det_cat_id,
            pred_instance.scores > self.bbox_thr)]

        if self.single_person:
            bboxes = self._select_primary(bboxes)

        pose_est_results = inference_topdown(self.pose_estimator, frame_bgr,
                                             bboxes)

        results_last = self._results_last
        converted = []
        for i, data_sample in enumerate(pose_est_results):
            pred_instances = data_sample.pred_instances.cpu().numpy()
            keypoints = pred_instances.keypoints
            if 'bboxes' in pred_instances:
                areas = np.array([(b[2] - b[0]) * (b[3] - b[1])
                                  for b in pred_instances.bboxes])
                pose_est_results[i].pred_instances.set_field(areas, 'areas')

            track_id, results_last, _ = self._track(data_sample, results_last,
                                                    self.tracking_thr)
            if track_id == -1:
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    track_id = self._next_id
                    self._next_id += 1
                else:
                    # too few keypoints detected: suppress this instance
                    keypoints[:, :, 1] = -10
                    pose_est_results[i].pred_instances.set_field(
                        keypoints, 'keypoints')
                    pose_est_results[i].pred_instances.set_field(
                        pred_instances.bboxes * 0, 'bboxes')
                    pose_est_results[i].set_field(pred_instances,
                                                  'pred_instances')
            pose_est_results[i].set_field(track_id, 'track_id')

            if not self.lifting_enabled:
                continue
            converted_sample = PoseDataSample()
            converted_sample.set_field(
                pose_est_results[i].pred_instances.clone(), 'pred_instances')
            converted_sample.set_field(pose_est_results[i].gt_instances.clone(),
                                       'gt_instances')
            kpts_converted = convert_keypoint_definition(
                keypoints, self.pose2d_dataset_name, self.pose3d_dataset_name)
            converted_sample.pred_instances.set_field(kpts_converted,
                                                      'keypoints')
            converted_sample.set_field(track_id, 'track_id')
            converted.append(converted_sample)

        self._results_last = pose_est_results

        if not self.lifting_enabled:
            return pose_est_results, []

        self._history.append(converted)
        max_history = max(1, self.seq_len * self.seq_step)
        if len(self._history) > max_history:
            del self._history[:-max_history]

        # Treat the newest frame as the target; extract_pose_sequence pads
        # the missing future frames by repeating the last one (online mode).
        pose_seq_2d = extract_pose_sequence(
            self._history,
            frame_idx=len(self._history) - 1,
            causal=self.causal,
            seq_len=self.seq_len,
            step=self.seq_step)

        pose_lift_results = inference_pose_lifter_model(
            self.pose_lifter,
            pose_seq_2d,
            image_size=frame_bgr.shape[:2],
            norm_pose_2d=self.norm_pose_2d)

        for idx, res in enumerate(pose_lift_results):
            res.track_id = pose_est_results[idx].get('track_id', 1e4)
            pred = res.pred_instances
            keypoint_scores = pred.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pred.keypoint_scores = keypoint_scores
            keypoints = pred.keypoints
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            # rotate to a right-handed Z-up frame for display (same
            # convention as the mmpose 3D visualizer)
            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]
            if self.rebase_keypoint:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)
            pred.keypoints = keypoints

        return pose_est_results, pose_lift_results


def _track_color(res):
    track_id = int(res.get('track_id', 0))
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


def log_2d(frame_bgr, pose_est_results, skeleton, kpt_thr, jpeg_quality):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rr.log('video/image', rr.Image(rgb).compress(jpeg_quality=jpeg_quality))

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
            'video/keypoints_2d',
            rr.Points2D(
                np.concatenate(points), colors=point_colors, radii=4))
    else:
        rr.log('video/keypoints_2d', rr.Clear(recursive=False))
    if strips:
        rr.log('video/skeleton_2d',
               rr.LineStrips2D(strips, colors=strip_colors, radii=2))
    else:
        rr.log('video/skeleton_2d', rr.Clear(recursive=False))
    if boxes:
        rr.log(
            'video/bboxes',
            rr.Boxes2D(
                array=np.stack(boxes),
                array_format=rr.Box2DFormat.XYXY,
                colors=[(220, 220, 220)]))
    else:
        rr.log('video/bboxes', rr.Clear(recursive=False))


def log_3d(pose_lift_results, skeleton, kpt_thr):
    points, point_colors = [], []
    strips, strip_colors = [], []
    for res in pose_lift_results:
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
            'pose3d/keypoints',
            rr.Points3D(
                np.concatenate(points), colors=point_colors, radii=0.02))
    else:
        rr.log('pose3d/keypoints', rr.Clear(recursive=False))
    if strips:
        rr.log('pose3d/skeleton',
               rr.LineStrips3D(strips, colors=strip_colors, radii=0.012))
    else:
        rr.log('pose3d/skeleton', rr.Clear(recursive=False))


def run_stream(cap, pipeline, cfg):
    skeleton_2d = pipeline.pose_estimator.dataset_meta.get(
        'skeleton_links') or []
    kpt_thr = float(cfg.get('kpt_thr', 0.3))
    jpeg_quality = int(cfg.get('rerun', {}).get('jpeg_quality', 75))

    skeleton_3d = None
    if pipeline.lifting_enabled:
        skeleton_3d = pipeline.pose_lifter.dataset_meta.get(
            'skeleton_links') or H36M_SKELETON
        rr.log('pose3d', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

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

        pose2d, pose3d = pipeline.step(frame)
        log_2d(frame, pose2d, skeleton_2d, kpt_thr, jpeg_quality)
        if pipeline.lifting_enabled:
            log_3d(pose3d, skeleton_3d, kpt_thr)

        fps_n += 1
        now = time.monotonic()
        if now - fps_t >= 5:
            print(f'frame {frame_idx}: {fps_n / (now - fps_t):.1f} FPS, '
                  f'{len(pose2d)} person(s)')
            fps_t, fps_n = now, 0


def smoke_test(cfg, device):
    """Exercise the full pipeline on synthetic frames, without camera/server."""
    rr.init('mmpose_body3d_smoke_test')
    pipeline = Body3DPipeline(cfg, device)
    skeleton_2d = pipeline.pose_estimator.dataset_meta.get(
        'skeleton_links') or []
    skeleton_3d = None
    if pipeline.lifting_enabled:
        skeleton_3d = pipeline.pose_lifter.dataset_meta.get(
            'skeleton_links') or H36M_SKELETON
    rng = np.random.default_rng(0)
    for idx in range(3):
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        rr.set_time_sequence('frame', idx)
        pose2d, pose3d = pipeline.step(frame)
        log_2d(frame, pose2d, skeleton_2d, 0.3, 75)
        if pipeline.lifting_enabled:
            log_3d(pose3d, skeleton_3d, 0.3)
        print(f'smoke test frame {idx}: {len(pose2d)} detection(s)')
    print(f'smoke test OK (lifting {"on" if pipeline.lifting_enabled else "off"})')


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
    views = [rrb.Spatial2DView(origin='video', name='Camera + 2D keypoints')]
    if cfg.get('lifting', {}).get('enabled', True):
        views.append(rrb.Spatial3DView(origin='pose3d', name='Lifted 3D pose'))
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
    pipeline = Body3DPipeline(cfg, device)
    cap = open_capture(cfg.get('camera', {}))
    print('Streaming. Press Ctrl+C to stop.')
    try:
        run_stream(cap, pipeline, cfg)
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
