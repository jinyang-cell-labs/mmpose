# Local deployment image for live body pose estimation (2D + lifted 3D)
# visualized with Rerun. Build from the repo root:
#   docker build -t mmpose-body3d-rerun -f local_deploy/Dockerfile .
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ffmpeg gcc g++ libgl1 libglib2.0-0 libsm6 libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# torch 2.1 / mmcv 2.1 are ABI-incompatible with numpy 2. Constrain every
# pip invocation in this image (including mim's) to numpy 1.x.
RUN printf 'numpy<2\n' > /opt/pip-constraints.txt
ENV PIP_CONSTRAINT=/opt/pip-constraints.txt

# OpenMMLab stack. mmcv 2.1.0 has prebuilt wheels for torch 2.1 + cu118;
# mmdet 3.2.0 is the newest release compatible with mmpose 1.3.x.
RUN pip install "numpy<2" pyyaml openmim cython "rerun-sdk==0.18.2" && \
    mim install "mmengine>=0.10.0" "mmcv==2.1.0" && \
    pip install "mmdet==3.2.0"

# mmpose runtime deps, installed explicitly instead of requirements/runtime.txt
# because `chumpy` (unused here) fails to build with modern pip/numpy.
# xtcocotools is compiled from source: its PyPI wheels are built against
# numpy 2 and crash on import under numpy 1.x.
RUN pip install json_tricks matplotlib munkres pillow scipy && \
    pip install --no-binary xtcocotools --no-build-isolation \
        "xtcocotools>=1.12"

WORKDIR /mmpose
COPY setup.py setup.cfg README.md model-index.yml dataset-index.yml ./
COPY requirements ./requirements
COPY mmpose ./mmpose
COPY configs ./configs
COPY demo/mmdetection_cfg ./demo/mmdetection_cfg
RUN pip install -e . --no-deps

# humanoid model for arm retargeting (visual meshes only, see .dockerignore)
COPY robot_model ./robot_model
COPY local_deploy ./local_deploy

# Rerun web viewer (HTTP) + websocket data stream
EXPOSE 9090 9877

CMD ["python", "local_deploy/app.py", "--config", "local_deploy/config.yaml"]
