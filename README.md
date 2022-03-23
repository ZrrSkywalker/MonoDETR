# MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
This repository is an official implementation of the paper ['MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection'](https://github.com/ZrrSkywalker/MonoDETR/blob/main/MonoDETR.pdf).

## Introduction
MonoDETR is the first DETR-based model for monocular 3D detection **without additional depth supervision, anchors or NMS**, which achieves leading performance on KITTI *val* and *test* set. Specifically, We enable the vanilla transformer to be depth-aware and enforce the whole detection process guided by depth. Specifically, we represent 3D object candidates as a set of queries and produce non-local depth embeddings of the input image by a lightweight depth predictor and an attention-based depth encoder. Then, we propose a depth-aware decoder to conduct both inter-query and query-scene depth feature communication. In this way, each object estimates its 3D attributes adaptively from the depth-informative regions on the image, not limited by center-around features.
<div align="center">
  <img src="pipeline.jpg"/ width="700px">
</div>



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


## Implementation
Comming soon!
