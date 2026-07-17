# Copyright (c) OpenMMLab. All rights reserved.
"""Stereo triangulation of 2D keypoints into metric 3D.

Calibration comes from two YAML files (see local_deploy/camera_params/):

- intrinsics.yaml: per camera, pinhole intrinsics [fx, fy, cx, cy] and
  radtan distortion [k1, k2, p1, p2] at a given calibration resolution.
- extrinsics.yaml: per camera, T_world_cam (4x4, camera pose in the world
  frame; the world frame is one of the cameras, typically camera0).

Points are undistorted to normalized image coordinates first, so the DLT
uses pure [R|t] projections and distortion is handled exactly.
"""
import cv2
import numpy as np
import yaml


class Camera:

    def __init__(self, name, K, dist, size, T_world_cam):
        self.name = name
        self.K = np.asarray(K, dtype=float)
        self.dist = np.asarray(dist, dtype=float)  # radtan: k1 k2 p1 p2
        self.size = tuple(int(v) for v in size) if size else None  # (w, h)
        self.T_world_cam = np.asarray(T_world_cam, dtype=float)
        # world -> normalized camera projection (intrinsics are applied by
        # undistorting the observations instead)
        self.P = np.linalg.inv(self.T_world_cam)[:3]

    def scale_to(self, width, height):
        """Adapt the intrinsics when capturing at a resolution other than
        the calibration one (distortion coefficients are resolution
        independent, only K scales)."""
        if self.size is None or (width, height) == self.size:
            return
        sx = width / self.size[0]
        sy = height / self.size[1]
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        self.K = K
        print(f'{self.name}: intrinsics scaled from calibration resolution '
              f'{self.size[0]}x{self.size[1]} to capture resolution '
              f'{width}x{height}')
        self.size = (width, height)

    def undistort(self, points_px):
        """(K, 2) pixel coords -> (K, 2) normalized image coords."""
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)


def load_stereo_rig(intrinsics_path, extrinsics_path):
    """Returns [Camera, Camera], ordered by camera name."""
    with open(intrinsics_path) as f:
        intr = yaml.safe_load(f)['cameras']
    with open(extrinsics_path) as f:
        extr = yaml.safe_load(f)['cameras']
    cams = []
    for name in sorted(intr):
        c = intr[name]
        fx, fy, cx, cy = c['intrinsics']
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        cams.append(
            Camera(name, K, c['distortion'], c.get('resolution'),
                   extr[name]['T_world_cam']))
    if len(cams) != 2:
        raise ValueError(f'Expected 2 cameras in {intrinsics_path}, '
                         f'got {len(cams)}: {sorted(intr)}')
    return cams


class StereoTriangulator:
    """Confidence-weighted linear triangulation (DLT) per keypoint."""

    def __init__(self, cams):
        self.cams = list(cams)

    def triangulate(self, kpts0, scores0, kpts1, scores1, kpt_thr=0.3):
        """Triangulate matched keypoint sets from the two views.

        A keypoint is triangulated only when its 2D score reaches kpt_thr
        in BOTH views; each view's DLT equations are weighted by its score.

        Returns (points (K, 3) in the world frame, valid (K,) bool,
        scores (K,) = min of the two views, zeroed where invalid).
        """
        kpts0 = np.asarray(kpts0, dtype=float)
        kpts1 = np.asarray(kpts1, dtype=float)
        scores0 = np.asarray(scores0, dtype=float)
        scores1 = np.asarray(scores1, dtype=float)
        n0 = self.cams[0].undistort(kpts0)
        n1 = self.cams[1].undistort(kpts1)

        n_kpts = len(kpts0)
        points = np.zeros((n_kpts, 3))
        valid = np.zeros(n_kpts, dtype=bool)
        for k in range(n_kpts):
            if scores0[k] < kpt_thr or scores1[k] < kpt_thr:
                continue
            rows = []
            for norm, score, cam in ((n0[k], scores0[k], self.cams[0]),
                                     (n1[k], scores1[k], self.cams[1])):
                rows.append(score * (norm[0] * cam.P[2] - cam.P[0]))
                rows.append(score * (norm[1] * cam.P[2] - cam.P[1]))
            _, _, vt = np.linalg.svd(np.stack(rows))
            hom = vt[-1]
            if abs(hom[3]) < 1e-12:
                continue
            p = hom[:3] / hom[3]
            # cheirality: the point must be in front of both cameras
            p_h = np.append(p, 1.0)
            if any((cam.P @ p_h)[2] <= 0 for cam in self.cams):
                continue
            points[k] = p
            valid[k] = True

        scores = np.minimum(scores0, scores1) * valid
        return points, valid, scores


def self_test():
    """Round-trip: project synthetic 3D points into two distorted cameras,
    triangulate them back, and compare."""
    K0 = [[900., 0, 640], [0, 905., 360], [0, 0, 1]]
    K1 = [[1100., 0, 620], [0, 1095., 380], [0, 0, 1]]
    d0 = [0.13, -0.21, 0.001, -0.006]
    d1 = [0.02, -0.17, 0.004, -0.0001]

    # camera1: 0.6 m to the right of camera0, yawed 25 deg toward the scene
    a = np.deg2rad(-25.0)
    T1 = np.eye(4)
    T1[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                  [-np.sin(a), 0, np.cos(a)]]
    T1[:3, 3] = [0.6, -0.1, 0.05]
    cams = [
        Camera('camera0', K0, d0, (1280, 720), np.eye(4)),
        Camera('camera1', K1, d1, (1280, 720), T1),
    ]

    rng = np.random.default_rng(0)
    pts_world = rng.uniform([-0.5, -0.8, 1.5], [0.5, 0.8, 3.0], size=(17, 3))

    def project(cam, pts):
        T_cam_world = np.linalg.inv(cam.T_world_cam)
        rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
        px, _ = cv2.projectPoints(pts, rvec, T_cam_world[:3, 3], cam.K,
                                  cam.dist)
        return px.reshape(-1, 2)

    px0, px1 = project(cams[0], pts_world), project(cams[1], pts_world)
    scores = np.ones(len(pts_world))
    scores[3] = 0.1  # below threshold in one view -> must come back invalid

    tri = StereoTriangulator(cams)
    rec, valid, out_scores = tri.triangulate(px0, scores, px1,
                                             np.ones(len(pts_world)), 0.3)
    assert not valid[3] and out_scores[3] == 0, 'score gating failed'
    err = np.linalg.norm(rec[valid] - pts_world[valid], axis=1)
    assert err.max() < 1e-6, f'triangulation error too high: {err.max()}'

    # intrinsics scaling must keep the round-trip exact
    half = Camera('camera0', K0, d0, (1280, 720), np.eye(4))
    half.scale_to(640, 360)
    tri2 = StereoTriangulator([half, cams[1]])
    rec2, valid2, _ = tri2.triangulate(px0 * 0.5, np.ones(len(pts_world)),
                                       px1, np.ones(len(pts_world)), 0.3)
    err2 = np.linalg.norm(rec2[valid2] - pts_world[valid2], axis=1)
    assert err2.max() < 1e-6, f'scaled triangulation error: {err2.max()}'
    print('triangulation self-test OK')


if __name__ == '__main__':
    self_test()
