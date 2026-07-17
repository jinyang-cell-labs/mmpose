# Copyright (c) OpenMMLab. All rights reserved.
"""Shared 2D stage: person detector -> top-down 2D pose -> tracking.

The heavy models are loaded once with load_models() and can be shared by
several Pose2DFrontend instances (one per camera); only the tracking /
primary-person state is per-instance.
"""
import numpy as np
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (_track_by_iou, _track_by_oks, inference_topdown,
                         init_model)
from mmpose.utils import adapt_mmdet_pipeline


def load_models(models_cfg, device):
    """Returns (detector, pose_estimator)."""
    detector = init_detector(
        models_cfg['det_config'], models_cfg['det_checkpoint'], device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_model(
        models_cfg['pose2d_config'],
        models_cfg['pose2d_checkpoint'],
        device=device)
    return detector, pose_estimator


class Pose2DFrontend:
    """Per-stream detection + 2D pose + tracking (+ single-person select).

    Keeps only per-frame state (last results for tracking, the selected
    primary bbox), so a long-running stream does not grow memory.
    """

    def __init__(self, detector, pose_estimator, cfg):
        self.detector = detector
        self.pose_estimator = pose_estimator

        det_cfg = cfg.get('detection', {})
        self.det_cat_id = int(det_cfg.get('cat_id', 0))
        self.bbox_thr = float(det_cfg.get('bbox_thr', 0.5))
        self.single_person = bool(det_cfg.get('single_person', False))
        self._selected_bbox = None  # last primary-person bbox (single_person)

        track_cfg = cfg.get('tracking', {})
        self.tracking_thr = float(track_cfg.get('thr', 0.3))
        self._track = _track_by_oks if track_cfg.get('use_oks',
                                                     False) else _track_by_iou

        self._results_last = []  # previous frame's 2D results, for tracking
        self._next_id = 0

    @property
    def dataset_name(self):
        return self.pose_estimator.dataset_meta['dataset_name']

    @property
    def skeleton(self):
        return self.pose_estimator.dataset_meta.get('skeleton_links') or []

    def _select_primary(self, bboxes):
        """Keep only the biggest person, with hysteresis toward the person
        selected in previous frames so the choice does not flip-flop between
        two similarly-sized people (which would corrupt any per-track 2D
        sequence downstream)."""
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
        """Process one frame. Returns pose_est_results with track ids set."""
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

        self._results_last = pose_est_results
        return pose_est_results
