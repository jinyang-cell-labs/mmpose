# Copyright (c) OpenMMLab. All rights reserved.
"""Retarget a lifted H36M 3D skeleton to humanoid arm joint angles, and
visualize the robot in Rerun via URDF forward kinematics.

Joint model per arm (matches robot_model/robot.urdf, zero pose = arms
hanging down along the torso):

    shoulder_pitch (about torso +Y, positive = arm backward)
      -> shoulder_roll (about +X, positive = left-arm abduction)
        -> shoulder_yaw (about the humerus axis +Z)
          -> elbow_pitch (about +Y, negative = flexion, forearm forward)

The upper-arm direction fixes pitch+roll; the forearm direction then fixes
yaw+elbow. All angles are clamped to the URDF limits.

The 3D keypoints from app.py live in the mmpose display frame (z-up), which
is mirrored (left-handed) w.r.t. the physical world; the torso axes below
are chosen so the derived angles are physically correct (validated by the
synthetic-pose self-test: `python retarget.py`).
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

# H36M keypoint indices
PELVIS, SPINE, THORAX = 0, 7, 8
L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16

ARM_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
              'elbow_pitch')

# below this elbow flexion the shoulder yaw is unobservable: hold last value
_YAW_HOLD_FLEXION = 0.26  # rad (~15 deg)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def torso_frame(kpts):
    """Rows are the torso axes in world coords: x=chest-forward, y=left,
    z=up. Returns None if the skeleton is degenerate."""
    z = _normalize(kpts[THORAX] - kpts[PELVIS])
    y = kpts[L_SHOULDER] - kpts[R_SHOULDER]
    if z is None:
        return None
    y = y - np.dot(y, z) * z  # make orthogonal to up
    y = _normalize(y)
    if y is None:
        return None
    # display frame is mirrored (det=-1), hence z cross y, not y cross z
    x = np.cross(z, y)
    return np.stack([x, y, z])


def arm_angles(kpts, side, prev_yaw=0.0):
    """Angles (pitch, roll, yaw, elbow) for one arm, or None if degenerate.

    kpts: (17, 3) H36M keypoints in the display frame.
    side: 'left' or 'right'.
    prev_yaw: yaw to hold when the elbow is too straight to observe it.
    """
    frame = torso_frame(kpts)
    if frame is None:
        return None
    sho, elb, wri = {
        'left': (L_SHOULDER, L_ELBOW, L_WRIST),
        'right': (R_SHOULDER, R_ELBOW, R_WRIST),
    }[side]

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


class ArmRetargeter:
    """Skeleton -> smoothed, limit-clamped joint angles for both arms."""

    def __init__(self, robot, smoothing=0.4, mirror=False):
        self.robot = robot
        self.alpha = float(smoothing)
        self.mirror = bool(mirror)
        self._state = {}  # joint name -> smoothed angle

    def update(self, kpts):
        """Returns {joint_name: angle} for all 8 arm joints."""
        out = {}
        for side in ('left', 'right'):
            src = side
            if self.mirror:
                src = 'right' if side == 'left' else 'left'
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


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(
        p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _axis_angle(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def _origin_transform(elem):
    t = np.eye(4)
    if elem is None:
        return t
    origin = elem.find('origin')
    if origin is None:
        return t
    xyz = [float(v) for v in (origin.get('xyz') or '0 0 0').split()]
    rpy = [float(v) for v in (origin.get('rpy') or '0 0 0').split()]
    t[:3, :3] = _rpy_matrix(*rpy)
    t[:3, 3] = xyz
    return t


class URDFRobot:
    """Minimal URDF loader: kinematic tree, joint limits, visual meshes,
    and forward kinematics. Supports revolute/fixed joints and STL visuals
    (which is all robot_model/robot.urdf uses)."""

    def __init__(self, urdf_path):
        self.dir = os.path.dirname(os.path.abspath(urdf_path))
        root = ET.parse(urdf_path).getroot()
        self.name = root.get('name', 'robot')

        self.joints = []  # dicts: name,type,parent,child,T_origin,axis
        self.limits = {}
        children = set()
        for j in root.iter('joint'):
            jtype = j.get('type')
            axis_el = j.find('axis')
            axis = [float(v) for v in axis_el.get('xyz').split()
                    ] if axis_el is not None else [1, 0, 0]
            info = dict(
                name=j.get('name'),
                type=jtype,
                parent=j.find('parent').get('link'),
                child=j.find('child').get('link'),
                T=_origin_transform(j),
                axis=axis)
            self.joints.append(info)
            children.add(info['child'])
            limit = j.find('limit')
            if jtype == 'revolute' and limit is not None:
                self.limits[info['name']] = (float(limit.get('lower')),
                                             float(limit.get('upper')))

        links = {l.get('name'): l for l in root.iter('link')}
        self.root_link = next(iter(set(links) - children))

        # visual mesh path + local origin per link
        self.visuals = {}  # link -> (abs_path, T_visual)
        for name, link in links.items():
            visual = link.find('visual')
            if visual is None:
                continue
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            fname = mesh.get('filename').replace('package://', '')
            self.visuals[name] = (os.path.join(self.dir, fname),
                                  _origin_transform(visual))

    def fk(self, joint_angles):
        """{joint_name: angle} -> {link_name: 4x4 pose in root frame}.
        Unspecified joints are at zero."""
        poses = {self.root_link: np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for j in pending:
                if j['parent'] not in poses:
                    rest.append(j)
                    continue
                t = poses[j['parent']] @ j['T']
                if j['type'] == 'revolute':
                    q = joint_angles.get(j['name'], 0.0)
                    rot = np.eye(4)
                    rot[:3, :3] = _axis_angle(j['axis'], q)
                    t = t @ rot
                poses[j['child']] = t
            if len(rest) == len(pending):
                raise ValueError('URDF kinematic tree is not connected')
            pending = rest
        return poses


def self_test():
    """Validate angle extraction on synthetic skeletons with known poses.

    The synthetic person is built directly in the display frame (z up,
    y away-from-camera, x camera-left) facing the camera, i.e. the person's
    left shoulder sits at -x, and their forward direction is -y.
    """
    def skeleton(l_elbow, l_wrist, r_elbow, r_wrist):
        k = np.zeros((17, 3))
        k[PELVIS] = (0, 0, 1.0)
        k[THORAX] = (0, 0, 1.5)
        k[L_SHOULDER] = (-0.2, 0, 1.45)
        k[R_SHOULDER] = (0.2, 0, 1.45)
        k[L_ELBOW] = k[L_SHOULDER] + l_elbow
        k[L_WRIST] = k[L_ELBOW] + l_wrist
        k[R_ELBOW] = k[R_SHOULDER] + r_elbow
        k[R_WRIST] = k[R_ELBOW] + r_wrist
        return k

    down, fwd = (0, 0, -0.25), (0, -0.25, 0)  # forward = toward camera = -y
    left_out, right_out = (-0.25, 0, 0), (0.25, 0, 0)

    # 1. arms hanging straight down -> all angles ~0
    k = skeleton(down, down, down, down)
    for side in ('left', 'right'):
        ang = arm_angles(k, side)
        assert all(abs(v) < 1e-6 for v in ang.values()), (side, ang)

    # 2. left arm straight forward -> pitch=-90deg, roll~0
    ang = arm_angles(k := skeleton(fwd, fwd, down, down), 'left')
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
    print('retarget self-test OK')


if __name__ == '__main__':
    self_test()
