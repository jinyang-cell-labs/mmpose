# Local Body Pose Deployment (Docker + Rerun)

Run live human body pose estimation on a local camera, fully containerized,
and visualize it in the [Rerun](https://rerun.io) viewer in your browser:

- **2D view** — the camera stream with the detected 17-keypoint skeletons,
  bounding boxes and per-person colors overlaid on the video.
- **3D view** — the lifted 3D pose (2D → 3D pose lifting), rendered as an
  interactive 3D skeleton you can orbit/zoom.

Pipeline (all monocular, single camera):

```
/dev/video0 ─> RTMDet-m (person detection)
            ─> RTMPose-m (2D keypoints, COCO-17)
            ─> VideoPose3D (2D→3D lifting, H36M-17)
            ─> Rerun web viewer (2D overlay + 3D plot)
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

## Validating the image without a camera

```bash
docker run --rm mmpose-body3d-rerun:latest \
    python local_deploy/app.py --config local_deploy/config.yaml --smoke-test
```

This runs the full detector → 2D → 3D pipeline on synthetic frames and exits.

## Notes & troubleshooting

- **The 3D pose is root-relative.** Monocular lifting cannot recover absolute
  metric position; the skeleton is centered at the pelvis and (by default)
  rebased so the lowest joint touches z = 0. For true metric 3D you'd need
  multi-camera triangulation of the 2D keypoints.
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
