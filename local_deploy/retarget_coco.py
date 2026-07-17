# Copyright (c) OpenMMLab. All rights reserved.
"""Retarget a triangulated COCO-17 3D skeleton to humanoid arm joint angles.

Counterpart of retarget.py, which handles the H36M skeleton produced by the
monocular lifter in the mmpose *display* frame (mirrored / left-handed).
The stereo pipeline instead yields metric keypoints in the COCO-17 format
in a proper right-handed z-up world, so this module differs in two ways:

- COCO indices; pelvis and thorax do not exist as joints and are derived
  as the hip / shoulder midpoints.
- The torso frame is built right-handed (x = y cross z), so the extracted
  angles are physically correct without the mirror trick.

The joint model per arm is identical to retarget.py (matches
robot_model/robot.urdf, zero pose = arms hanging down along the torso):

    shoulder_pitch (about torso +Y, positive = arm backward)
      -> shoulder_roll (about +X, positive = left-arm abduction)
        -> shoulder_yaw (about the humerus axis +Z)
          -> elbow_pitch (about +Y, negative = flexion, forearm forward)

Validated by the synthetic-pose self-test: `python retarget_coco.py`.
"""
import numpy as np

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

ARM_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
              'elbow_pitch')

_SIDE_JOINTS = {
    'left': (L_SHOULDER, L_ELBOW, L_WRIST),
    'right': (R_SHOULDER, R_ELBOW, R_WRIST),
}
_TORSO_JOINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

# below this elbow flexion the shoulder yaw is unobservable: hold last value
_YAW_HOLD_FLEXION = 0.26  # rad (~15 deg)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def torso_frame(kpts):
    """Rows are the torso axes in world coords: x=chest-forward, y=left,
    z=up (right-handed). Returns None if the skeleton is degenerate."""
    pelvis = (kpts[L_HIP] + kpts[R_HIP]) / 2
    thorax = (kpts[L_SHOULDER] + kpts[R_SHOULDER]) / 2
    z = _normalize(thorax - pelvis)
    if z is None:
        return None
    y = kpts[L_SHOULDER] - kpts[R_SHOULDER]
    y = y - np.dot(y, z) * z  # make orthogonal to up
    y = _normalize(y)
    if y is None:
        return None
    x = np.cross(y, z)  # right-handed: left x up = forward
    return np.stack([x, y, z])


def arm_angles(kpts, side, prev_yaw=0.0):
    """Angles (pitch, roll, yaw, elbow) for one arm, or None if degenerate.

    kpts: (17, 3) COCO keypoints in a right-handed z-up world frame.
    side: 'left' or 'right'.
    prev_yaw: yaw to hold when the elbow is too straight to observe it.
    """
    frame = torso_frame(kpts)
    if frame is None:
        return None
    sho, elb, wri = _SIDE_JOINTS[side]

    a = _normalize(frame @ (kpts[elb] - kpts[sho]))  # upper arm, torso frame
    f = _normalize(frame @ (kpts[wri] - kpts[elb]))  # forearm, torso frame
    if a is None or f is None:
        return None

    # upper arm at zero pose is (0,0,-1); a = Ry(pitch) @ Rx(roll) @ (0,0,-1)
    roll = np.arcsin(np.clip(a[1], -1.0, 1.0))
    pitch = np.arctan2(-a[0], -a[2])

    # undo pitch/roll, then f_local = Rz(yaw) @ Ry(elbow) @ (0,0,-1)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry_t = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])  # Ry(pitch).T
    rx_t = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])  # Rx(roll).T
    fl = rx_t @ (ry_t @ f)

    elbow = -np.arccos(np.clip(-fl[2], -1.0, 1.0))  # flexion is negative
    if abs(elbow) > _YAW_HOLD_FLEXION:
        yaw = np.arctan2(fl[1], fl[0])
    else:
        yaw = prev_yaw
    return {
        'shoulder_pitch': float(pitch),
        'shoulder_roll': float(roll),
        'shoulder_yaw': float(yaw),
        'elbow_pitch': float(elbow),
    }


class CocoArmRetargeter:
    """COCO skeleton -> smoothed, limit-clamped joint angles for both arms.

    update() additionally takes the per-keypoint validity mask from the
    triangulation, so arms with untriangulated joints are simply skipped
    (they keep their last smoothed angles).
    """

    def __init__(self, robot, smoothing=0.4, mirror=False):
        self.robot = robot
        self.alpha = float(smoothing)
        self.mirror = bool(mirror)
        self._state = {}  # joint name -> smoothed angle

    def update(self, kpts, valid=None):
        """Returns {joint_name: angle} for the arms that could be solved."""
        out = {}
        if valid is not None and not all(valid[j] for j in _TORSO_JOINTS):
            return out
        for side in ('left', 'right'):
            src = side
            if self.mirror:
                src = 'right' if side == 'left' else 'left'
            if valid is not None and not all(
                    valid[j] for j in _SIDE_JOINTS[src]):
                continue
            prev_yaw = self._state.get(f'shoulder_yaw_{side}', 0.0)
            ang = arm_angles(kpts, src, prev_yaw=prev_yaw)
            if ang is None:
                continue
            if self.mirror:  # mirror symmetry flips roll and yaw
                ang['shoulder_roll'] = -ang['shoulder_roll']
                ang['shoulder_yaw'] = -ang['shoulder_yaw']
            for j, q in ang.items():
                name = f'{j}_{side}'
                lo, hi = self.robot.limits.get(name, (-np.pi, np.pi))
                q = float(np.clip(q, lo, hi))
                prev = self._state.get(name)
                if prev is not None:
                    q = self.alpha * q + (1 - self.alpha) * prev
                self._state[name] = q
                out[name] = q
        return out


def self_test():
    """Validate angle extraction on synthetic skeletons with known poses.

    The synthetic person stands in a right-handed z-up world facing +x,
    i.e. their left side points along +y.
    """

    def skeleton(l_elbow, l_wrist, r_elbow, r_wrist):
        k = np.zeros((17, 3))
        k[L_HIP] = (0, 0.12, 1.0)
        k[R_HIP] = (0, -0.12, 1.0)
        k[L_SHOULDER] = (0, 0.2, 1.45)
        k[R_SHOULDER] = (0, -0.2, 1.45)
        k[L_ELBOW] = k[L_SHOULDER] + l_elbow
        k[L_WRIST] = k[L_ELBOW] + l_wrist
        k[R_ELBOW] = k[R_SHOULDER] + r_elbow
        k[R_WRIST] = k[R_ELBOW] + r_wrist
        return k

    down, fwd = (0, 0, -0.25), (0.25, 0, 0)  # forward = +x
    left_out, right_out = (0, 0.25, 0), (0, -0.25, 0)

    # 0. torso frame is right-handed: forward must be +x
    frame = torso_frame(skeleton(down, down, down, down))
    assert np.allclose(frame[0], (1, 0, 0)), frame

    # 1. arms hanging straight down -> all angles ~0
    k = skeleton(down, down, down, down)
    for side in ('left', 'right'):
        ang = arm_angles(k, side)
        assert all(abs(v) < 1e-6 for v in ang.values()), (side, ang)

    # 2. left arm straight forward -> pitch=-90deg, roll~0
    ang = arm_angles(skeleton(fwd, fwd, down, down), 'left')
    assert abs(ang['shoulder_pitch'] + np.pi / 2) < 1e-6, ang
    assert abs(ang['shoulder_roll']) < 1e-6, ang

    # 3. arms abducted sideways -> left roll=+90deg, right roll=-90deg
    ang_l = arm_angles(skeleton(left_out, left_out, down, down), 'left')
    ang_r = arm_angles(skeleton(down, down, right_out, right_out), 'right')
    assert abs(ang_l['shoulder_roll'] - np.pi / 2) < 1e-6, ang_l
    assert abs(ang_r['shoulder_roll'] + np.pi / 2) < 1e-6, ang_r

    # 4. arm hanging, elbow flexed forward -> elbow=-90deg, yaw~0
    ang = arm_angles(skeleton(down, fwd, down, down), 'left')
    assert abs(ang['elbow_pitch'] + np.pi / 2) < 1e-6, ang
    assert abs(ang['shoulder_yaw']) < 1e-6, ang

    # 5. validity gating: an untriangulated wrist drops that arm only
    class _Robot:
        limits = {}

    ret = CocoArmRetargeter(_Robot(), smoothing=1.0)
    valid = np.ones(17, dtype=bool)
    valid[L_WRIST] = False
    out = ret.update(skeleton(down, down, down, down), valid)
    assert not any(n.endswith('_left') for n in out), out
    assert any(n.endswith('_right') for n in out), out
    print('retarget_coco self-test OK')


if __name__ == '__main__':
    self_test()
