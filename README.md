# CVPR'25 Hand-held Object Reconstruction from RGB Video with Dynamic Interaction
This is the official repo for the implementation of **Hand-held Object Reconstruction from RGB Video with Dynamic Interaction**.  
Shijian Jiang, Qi Ye, Rengan Xie, Yuchi Huo, Jiming Chen
## [Project page](https://east-j.github.io/dynhor/) |  [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Hand-held_Object_Reconstruction_from_RGB_Video_with_Dynamic_Interaction_CVPR_2025_paper.html) | [Data](https://drive.google.com/drive/folders/1q6KSatlFLYWkqny4_aS8w5S_hSp-Jlc4?usp=sharing)
<img src="assets/shoes_res.gif" width="300"> <img src="assets/kettle_res.gif" width="300">

## TODO List
- [x] ✅ Release the object pose estimation code
- [ ] 🛠️ Release the processing code with custom data 
- [ ] 🛠️ Release the reconstruction code based on NeuS  
- [ ] 🚀 Replace NeuS with instant-nsr-pl for faster reconstruction
      
🚧**[WIP]**: I’ve updated the code in the `dev` branch to include the reconstruction part using [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl). You can follow the instructions in `example.sh` to use it. However, some parts of the code may still require further adjustments.

## Installation
### Set up the environment
```bash
conda create -n dynhor python=3.10
# Pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# requirements
pip install -r ObjTracker/requirements.txt
```

## Usage

#### Data Convention
The data is organized as follows:
```
<seq_name>
|-- rgb
    |-- 0000.png        # target image for each view
    |-- 0001.png
    ...
|-- sam_seg
    |-- 0000.png        # segmentation for each view obtained using SAM-v2
    |-- 0001.png
    ...
| -- monocular_normal   
    |-- 0000.png        # monocular normal for each view obtained using StableNormal
    |-- 0001.png
    ...
| -- correspondence_infos # dense correspondence obtained using DKM for reconstruction and outlier-voting
```
You can download the demo data from [here](https://drive.google.com/drive/folders/1q6KSatlFLYWkqny4_aS8w5S_hSp-Jlc4?usp=sharing).

#### Running
- **Estimate object poses**
```bash
cd ./ObjTracker
python run.py --config_path ./configs/custom_shoes.yaml 
# After running, you can render the results
python vis.py --config_path ./exps/custom_shoes/pred/custom_shoes.yaml 
```
- **Reconstruct object**
```bash
cd ../NeuS
```

## Citation
Cite as below if you find this repository is helpful to your project:
```
@inproceedings{jiang2025hand,
  title={Hand-held Object Reconstruction from RGB Video with Dynamic Interaction},
  author={Jiang, Shijian and Ye, Qi and Xie, Rengan and Huo, Yuchi and Chen, Jiming},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={12220--12230},
  year={2025}
}
```

## Acknowledgments

Our code benefits a lot from [homan](https://github.com/hassony2/homan), [NeuS](https://github.com/Totoro97/NeuS), [HHOR](https://github.com/dihuangdh/HHOR). If you find our work useful, consider checking out their work.
