# MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
Official implementation of the paper ['MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection'](https://github.com/ZrrSkywalker/MonoDETR/blob/main/MonoDETR.pdf).

## Introduction
MonoDETR is the first DETR-based model for monocular 3D detection **without additional depth supervision, anchors or NMS**, which achieves leading performance on KITTI *val* and *test* set. We enable the vanilla transformer in DETR to be depth-aware and enforce the whole detection process guided by depth. In this way, each object estimates its 3D attributes adaptively from the depth-informative regions on the image, not limited by center-around features.
<div align="center">
  <img src="pipeline.jpg"/>
</div>

## Implementation
We provide the checkpoints trained on KITTI *train* set with the performance on *val* set of Car AP<sub>R40</sub> as:
<table>
    <tr>
        <td div align="center"></td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td>  
    </tr>
    <tr>
        <td div align="center">In the paper</td>
        <td div align="center">26.66%</td> 
        <td div align="center">20.14%</td> 
        <td div align="center">16.88%</td> 
    </tr>
    <tr>
        <td div align="center">In this repo</td>
        <td div align="center">28.92%</td> 
        <td div align="center">20.74%</td> 
        <td div align="center">17.20%</td> 
    </tr>
</table>


## Installation
Comming soon!

## Acknowlegment
This repo benefits from the excellent [MonoDLE](https://github.com/xinzhuma/monodle) and [GUPNet](https://github.com/SuperMHP/GUPNet).

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
