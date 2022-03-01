# MonoDETR: Depth-aware Monocular Detection Transformer
This repository is an official implementation of the paper of 'Depth-aware Monocular Detection Transformer'. The paper will be arxiv available soon.

<table align="center">
    <tr>
        <td colspan="3",div align="center">KITTI valset, Car @IoU=0.7, AP_R40</td>    
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td>  
    </tr>
    <tr>
        <td div align="center">28.54%</td> 
        <td div align="center">20.55%</td> 
        <td div align="center">16.88%</td> 
    </tr>
</table>

## Introduction
MonoDETR is the first DETR-based model for monocular 3D detection, which achieves state-of-the-art performance on KITTI dataset.  Specifically, we abandon the sub-optimal center-driven paradigms adopted by existing detectors, but predict 3D bounding boxes from adaptive depth-guided regions. We encode scene-level depth embeddings under the supervision of a constructed pseudo depth map, and design a depth-aware cross-attention module for object-scene depth interactions.
<div align="center">
  <img src="pipeline.jpg"/>
</div>



## Implementation
Comming soon!
