# Copyright (c) OpenMMLab. All rights reserved.
"""Shared building blocks for the local deployment entrypoints.

- app.py        : mono pipeline (detector -> 2D pose -> 2D->3D lifting)
- app_stereo.py : stereo pipeline (2x [detector -> 2D pose] -> triangulation)
"""
