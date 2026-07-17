# Local Body Pose Deployment (Docker + Rerun)

Run live human body pose estimation on a local camera, fully containerized,
and visualize it in the [Rerun](https://rerun.io) viewer in your browser:

- **2D view** — the camera stream with the detected 17-keypoint skeletons,
  bounding boxes and per-person colors overlaid on the video.
- **3D view** — the lifted 3D pose (2D → 3D pose lifting), rendered as an
  interactive 3D skeleton you can orbit/zoom.

Two pipelines share the same image and code base
(`local_deploy/core/` holds the shared stages):

**Mono** (default, `app.py` / `config.yaml`) — single camera, 3D by
2D→3D lifting (root-relative, not metric):

```
/dev/video0 ─> RTMDet-m (person detection)
            ─> RTMPose-m (2D keypoints, COCO-17)
            ─> VideoPose3D (2D→3D lifting, H36M-17)
            ─> Rerun web viewer (2D overlay + 3D plot)
```

**Stereo** (`app_stereo.py` / `config_stereo.yaml`) — two calibrated
cameras, metric 3D by triangulation (see
[Stereo pipeline](#stereo-pipeline-metric-3d-from-two-cameras)):

```
/dev/video0 ─┐
             ├─> RTMDet-m + RTMPose-m (per camera, shared models)
/dev/video2 ─┘
             ─> triangulation (calibration in camera_params/)
             ─> metric 3D skeleton (COCO-17) ─> arm retargeting
             ─> Rerun web viewer (2 camera views + 3D plot)
```

## Prerequisites

- Docker (a GPU is used automatically if the NVIDIA container runtime is
  installed; otherwise it falls back to CPU with low FPS).
- A V4L2 camera at `/dev/video0` (configurable, see below) — or a video file.
- ~12 GB free disk for the image; model checkpoints (~400 MB) are downloaded
  on first start and cached in a Docker volume.

## Quick start

From the repo root:

```bash
./local_deploy/bootstrap.sh
```

This builds the image (first build takes a while), starts the container with
the camera passed through, and prints the viewer URL. Then open:

**http://localhost:9090?url=ws://localhost:9877**

You should see the camera stream with the 2D skeleton overlay on the left and
the lifted 3D pose on the right. Use the timeline at the bottom to scrub back
through past frames.

Stop with `Ctrl+C`. Re-run without rebuilding:

```bash
SKIP_BUILD=1 ./local_deploy/bootstrap.sh
```

## Configuration

Everything is configured in [`config.yaml`](config.yaml). The file is
bind-mounted into the container, so **edits only require a container restart,
not a rebuild**.

| Key | Default | Meaning |
| --- | --- | --- |
| `camera.path` | `/dev/video0` | Video source: V4L2 device path, bare index (`0` = `/dev/video0`), or a video file path. `bootstrap.sh` reads this to pass the device/file into the container. |
| `camera.width` / `camera.height` | 640 / 480 | Requested capture resolution (optional). |
| `device` | `auto` | `auto` = CUDA if available, else CPU. Can force `cpu` or `cuda:0`. |
| `models.*` | RTMDet-m / RTMPose-m / VideoPose3D | Config + checkpoint (URL or path) for each stage. |
| `detection.bbox_thr` | 0.5 | Min person-detection score. |
| `detection.single_person` | true | Keep only the biggest person in frame (with hysteresis so the choice doesn't flip between similar-sized people). Background people are dropped before the 2D stage, so they cost nothing and never pollute the 3D lifting history. Set `false` for multi-person. |
| `tracking.use_oks` | false | Track people across frames by IoU (false) or OKS (true). Tracking feeds the per-person 2D sequence into the lifter. |
| `lifting.enabled` | true | Set to `false` to skip the 3D lifting stage entirely (the lifter model is not loaded): faster, less GPU memory, and the viewer shows only the 2D overlay. |
| `lifting.norm_pose_2d` | true | Normalize bbox/pose to dataset stats before lifting (helps small/far people). |
| `lifting.rebase_keypoint` | true | Put the lowest joint of each 3D pose at z = 0 (the lifter predicts root-relative coordinates, not global position). |
| `kpt_thr` | 0.3 | Min keypoint score to draw. |
| `robot.enabled` | true | Retarget the lifted 3D pose to the humanoid arms (`robot_model/robot.urdf`) and render the robot next to the skeleton, with joint-angle plots. Needs `lifting.enabled`. |
| `robot.smoothing` | 0.35 | EMA factor for the joint angles (1 = raw). |
| `robot.mirror` | false | Swap left/right arms if webcam mirroring makes the robot follow the wrong side. |
| `robot.offset` / `robot.yaw_deg` | `[0,-0.9,0.95]` / 0 | Robot placement in the 3D view. |
| `rerun.web_port` / `rerun.ws_port` | 9090 / 9877 | Viewer HTTP port / data websocket port (bootstrap publishes both). |

### Using a different camera

Set `camera.path` to e.g. `/dev/video2` and restart. `bootstrap.sh` picks the
device up from the config and passes it through with `--device`.

### Using a video file instead of a camera

Set `camera.path` to an absolute path of a video file on the host;
`bootstrap.sh` mounts it read-only into the container. When the file ends, the
container keeps the Rerun server alive so you can keep inspecting the
recording.

### Swapping models

Any top-down 2D config from `configs/body_2d_keypoint/` and any lifter from
`configs/body_3d_keypoint/` works — the configs directory is baked into the
image. E.g. for MotionBERT (better 3D quality, heavier), set:

```yaml
  pose3d_config: configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py
  pose3d_checkpoint: https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth
```

## Stereo pipeline (metric 3D from two cameras)

```bash
STEREO=1 ./local_deploy/bootstrap.sh
```

This runs `app_stereo.py` with [`config_stereo.yaml`](config_stereo.yaml):
both cameras are passed through, the per-view 2D keypoints are triangulated
with the calibration in [`camera_params/`](camera_params/), and the viewer
shows both camera streams, the metric 3D skeleton, the camera frustums and
the retargeted robot.

Key differences from the mono pipeline:

- **Metric & absolute 3D.** Triangulation recovers true positions in the
  calibrated world frame (= `camera0`, remapped to z-up for display), so
  there is no root-relative/`rebase_keypoint` approximation.
- **COCO-17 end to end.** No pose lifter and no H36M conversion; the
  triangulated skeleton stays in the COCO format and is retargeted by
  [`retarget_coco.py`](retarget_coco.py) (pelvis/thorax derived as
  hip/shoulder midpoints; right-handed frame, unlike the mirrored display
  frame the mono `retarget.py` compensates for). Self-test:
  `python local_deploy/retarget_coco.py`.
- **Single person.** Cross-view matching uses the primary (biggest, sticky)
  person of each view; `detection.single_person` is forced on.
- **Both frames are `grab()`bed back-to-back** before decoding, bounding the
  view skew to a few ms without hardware sync.
- Expect roughly half the FPS of mono: the 2D stage runs twice per frame
  pair on the same GPU.

Stereo-specific configuration (`config_stereo.yaml`):

| Key | Default | Meaning |
| --- | --- | --- |
| `cameras.cam0` / `cameras.cam1` | `/dev/video0` / `/dev/video2` | Per-view capture settings (same options as mono `camera.*`). `cam0`/`cam1` must be the rig's `camera0`/`camera1` from the calibration files. |
| `calibration.intrinsics` / `.extrinsics` | `camera_params/*.yaml` | Pinhole + radtan intrinsics per camera; `T_world_cam` extrinsics (world = `camera0`). Capture at the calibration resolution if possible — other resolutions are handled by rescaling the intrinsics. |
| `triangulation.kpt_thr` | 0.35 | A keypoint is triangulated only if its 2D score reaches this in **both** views (the DLT is also confidence-weighted; points behind a camera are rejected). |
| `viz.yaw_deg` / `viz.translation` | 0 / `[0,0,0]` | Extra transform applied to the z-up display frame, e.g. to put the floor at z = 0 (depends on the height of `camera0`). |
| `robot.mirror` | false | The stereo rig sees the non-mirrored world, so the robot follows the correct side by default. |

The geometry self-tests (undistort → weighted-DLT round-trip, COCO
retargeter) run as part of `--smoke-test`, or standalone:
`python local_deploy/core/triangulation.py`.

## Validating the image without a camera

```bash
docker run --rm mmpose-body3d-rerun:latest \
    python local_deploy/app.py --config local_deploy/config.yaml --smoke-test
docker run --rm mmpose-body3d-rerun:latest \
    python local_deploy/app_stereo.py --config local_deploy/config_stereo.yaml --smoke-test
```

This runs the full detector → 2D → 3D pipeline (lifting resp. triangulation)
on synthetic frames and exits.

## Robot arm retargeting

With `robot.enabled: true` the lifted H36M skeleton drives the humanoid in
`robot_model/robot.urdf`: per arm, **shoulder pitch / roll / yaw + elbow
pitch** are computed analytically from the upper-arm and forearm directions
expressed in a torso frame (up = pelvis→thorax, left = right→left shoulder),
using the robot's convention (zero pose = arms hanging, pitch about Y, roll
about X, yaw about the humerus axis, elbow about Y). Angles are clamped to
the URDF limits, smoothed with an EMA, and the shoulder yaw is held when the
elbow is near-straight (it is geometrically unobservable then). The robot is
rendered by forward kinematics in the same 3D view as the skeleton
(`retarget.py`, self-tested against synthetic poses — run
`python local_deploy/retarget.py` to check). Wrist/leg/head joints stay at
zero. Note the URDF's true joint axes are slightly tilted from the nominal
pitch/roll/yaw axes (actuator packaging), so the mesh pose can deviate a few
degrees from the human pose.

## Notes & troubleshooting

- **The mono 3D pose is root-relative.** Monocular lifting cannot recover
  absolute metric position; the skeleton is centered at the pelvis and (by
  default) rebased so the lowest joint touches z = 0. For true metric 3D use
  the stereo pipeline (`STEREO=1`, see above).
- **Latency/quality trade-off:** the VideoPose3D lifter uses a 243-frame
  temporal window. In this live setup the newest frame is the target and the
  future half of the window is padded, so the 3D pose can lag or wobble
  slightly on fast motion; it stabilizes as history accumulates.
- **Camera cannot be opened:** check the device exists (`ls /dev/video*`),
  is not in use, and that your user can access it (typically the `video`
  group). The container runs as root, so host-side permissions rarely matter,
  but the `--device` flag must have been applied (see bootstrap output).
- **Ports already in use:** change `rerun.web_port`/`rerun.ws_port` in
  `config.yaml`; bootstrap reads them for the `-p` mappings.
- **First start is slow:** ~400 MB of checkpoints are downloaded once into
  the `mmpose-checkpoints` Docker volume and reused afterwards
  (`docker volume rm mmpose-checkpoints` to clear).
- **No GPU picked up:** install `nvidia-container-toolkit` and restart the
  Docker daemon; bootstrap auto-detects it and adds `--gpus all`.
