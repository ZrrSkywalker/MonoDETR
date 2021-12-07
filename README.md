# MonoDETR: Depth-aware Monocular Detection Transformer
This repository is an official implementation of the paper of 'Depth-aware Monocular Detection Transformer'.

## Introduction
MonoDETR is the first DETR-based model for monocular 3D detection, which achieves state-of-the-art performance on KITTI dataset.  Specifically, we abandon the sub-optimal center-driven paradigms adopted by existing detectors, but predict 3D bounding boxes from adaptive depth-guided regions. We encode scene-level depth embeddings under the supervision of a constructed pseudo depth map, and design a depth-aware cross-attention module for object-scene depth interactions.
