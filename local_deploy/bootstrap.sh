#!/usr/bin/env bash
# Bootstrap for the local body-pose deployment: builds the Docker image and
# starts the container with the camera(s) (or video files) from the config
# passed through, and the Rerun viewer ports published.
#
# Usage:
#   ./local_deploy/bootstrap.sh              # mono pipeline (config.yaml)
#   STEREO=1 ./local_deploy/bootstrap.sh     # stereo pipeline (config_stereo.yaml)
#   SKIP_BUILD=1 ./local_deploy/bootstrap.sh # run only, skip docker build
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
IMAGE=${IMAGE:-mmpose-body3d-rerun:latest}
STEREO=${STEREO:-0}

if [[ "$STEREO" == "1" ]]; then
    CONFIG="$SCRIPT_DIR/config_stereo.yaml"
    APP_CMD=(python local_deploy/app_stereo.py
             --config local_deploy/config_stereo.yaml)
else
    CONFIG="$SCRIPT_DIR/config.yaml"
    APP_CMD=()  # image default: local_deploy/app.py
fi

yaml_value() { # yaml_value <section> <key> -> first "key:" value inside section
    awk -v section="$1" -v key="$2" '
        $0 ~ "^" section ":" {in_section=1; next}
        in_section && /^[^[:space:]#]/ {in_section=0}
        in_section && $1 == key ":" {print $2; exit}
    ' "$CONFIG"
}

# Every "path:" key in the config is a video source (one for mono, two for
# stereo; the calibration files use other key names).
CAM_PATHS=$(awk '$1 == "path:" {print $2}' "$CONFIG")
[[ -n "$CAM_PATHS" ]] || CAM_PATHS="/dev/video0"
WEB_PORT=$(yaml_value rerun web_port); WEB_PORT=${WEB_PORT:-9090}
WS_PORT=$(yaml_value rerun ws_port); WS_PORT=${WS_PORT:-9877}

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "==> Building image $IMAGE (context: $REPO_ROOT)"
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
fi

RUN_ARGS=(
    --rm -i
    -p "$WEB_PORT:$WEB_PORT" -p "$WS_PORT:$WS_PORT"
    # persist downloaded checkpoints across runs
    -v mmpose-checkpoints:/root/.cache
    # live-mount the config so edits only need a container restart
    -v "$CONFIG":/mmpose/local_deploy/$(basename "$CONFIG"):ro
)
[[ -t 0 ]] && RUN_ARGS+=(-t)

if [[ "$STEREO" == "1" ]]; then
    # live-mount the calibration too (edits only need a restart)
    RUN_ARGS+=(-v "$SCRIPT_DIR/camera_params":/mmpose/local_deploy/camera_params:ro)
fi

# GPU passthrough when the NVIDIA container runtime is available
if command -v nvidia-smi >/dev/null 2>&1 \
        && docker info 2>/dev/null | grep -qi nvidia; then
    echo "==> NVIDIA runtime detected: enabling GPU"
    RUN_ARGS+=(--gpus all)
else
    echo "==> No NVIDIA runtime detected: running on CPU (expect low FPS)"
fi

# Camera device or video file passthrough (one per "path:" in the config)
for CAM_PATH in $CAM_PATHS; do
    # Bare index like "0" means /dev/video0 on the host side
    if [[ "$CAM_PATH" =~ ^[0-9]+$ ]]; then
        HOST_CAM="/dev/video$CAM_PATH"
    else
        HOST_CAM="$CAM_PATH"
    fi
    if [[ -c "$HOST_CAM" ]]; then
        echo "==> Passing camera device $HOST_CAM into the container"
        RUN_ARGS+=(--device "$HOST_CAM:$HOST_CAM")
    elif [[ -f "$HOST_CAM" ]]; then
        echo "==> Mounting video file $HOST_CAM into the container"
        RUN_ARGS+=(-v "$HOST_CAM":"$HOST_CAM":ro)
    else
        echo "WARNING: camera path '$HOST_CAM' not found on host." >&2
        echo "         Edit the path in $CONFIG (available: $(ls /dev/video* 2>/dev/null || echo none))" >&2
    fi
done

echo
echo "==> Open the Rerun viewer at: http://localhost:$WEB_PORT?url=ws://localhost:$WS_PORT"
echo "    (models are downloaded on first start; give it a minute)"
echo
exec docker run "${RUN_ARGS[@]}" "$@" "$IMAGE" ${APP_CMD[@]+"${APP_CMD[@]}"}
