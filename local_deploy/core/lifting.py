# Copyright (c) OpenMMLab. All rights reserved.
"""Monocular 2D->3D stage: temporal pose lifting (VideoPose3D et al.).

Extracted from the original Body3DPipeline; keeps only a bounded history of
2D results so a long-running webcam stream does not grow memory without
limit.
"""
import numpy as np

from mmpose.apis import (convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, init_model)
from mmpose.structures import PoseDataSample


class TemporalLifter:
    """Per-track 2D keypoint sequences -> root-relative 3D pose."""

    def __init__(self, cfg, pose2d_dataset_name, device):
        models = cfg['models']
        lift_cfg = cfg.get('lifting', {})
        self.pose_lifter = init_model(
            models['pose3d_config'], models['pose3d_checkpoint'],
            device=device)

        lift_dataset = self.pose_lifter.cfg.test_dataloader.dataset
        self.seq_len = lift_dataset.get('seq_len', 1)
        self.seq_step = lift_dataset.get('seq_step', 1)
        self.causal = lift_dataset.get('causal', False)

        self.norm_pose_2d = bool(lift_cfg.get('norm_pose_2d', True))
        self.rebase_keypoint = bool(lift_cfg.get('rebase_keypoint', True))
        self.pose2d_dataset_name = pose2d_dataset_name
        self.pose3d_dataset_name = self.pose_lifter.dataset_meta[
            'dataset_name']

        self._history = []  # converted 2D results of the last N frames

    @property
    def dataset_meta(self):
        return self.pose_lifter.dataset_meta

    def step(self, pose_est_results, image_size):
        """Lift the tracked 2D results of one frame. Returns 3D results in
        the mmpose display frame (right-handed z-up, mirrored)."""
        converted = []
        for res in pose_est_results:
            converted_sample = PoseDataSample()
            converted_sample.set_field(res.pred_instances.clone(),
                                       'pred_instances')
            converted_sample.set_field(res.gt_instances.clone(),
                                       'gt_instances')
            keypoints = np.asarray(res.pred_instances.keypoints)
            kpts_converted = convert_keypoint_definition(
                keypoints, self.pose2d_dataset_name, self.pose3d_dataset_name)
            converted_sample.pred_instances.set_field(kpts_converted,
                                                      'keypoints')
            converted_sample.set_field(res.get('track_id', -1), 'track_id')
            converted.append(converted_sample)

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
            image_size=image_size,
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

        return pose_lift_results
