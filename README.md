# ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation
The official implementation of the paper: [**ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation**](https://arxiv.org/abs/2401.14074)
![TEL](image/framework.png)

# 🔔News
- 2024-12-23, 🎉🎉 Our paper "[**ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation**](https://arxiv.org/abs/2401.14074)" has been accepted by **IEEE Journal of Biomedical and Health Informatics (JBHI)**.
  

# Datasets
## Download the Processed Datasets (N/A will be included in subsequent updates.)

<table>
  <tbody>
    <tr>
      <td align="center"></td>
      <td align="center">Fundus</td>
      <td align="center">OCTA</td>
      <td align="center">Endoscopy</td>
      <td align="center">Cardiac MRI</td>
      <td align="center">Brain Tumor MRI</td>
      <td align="center">H\&E</td>
    </tr>
    <tr>
      <td align="Sparse annotation"></td>
      <td align="center">scribble</td>
      <td align="center">point</td>
      <td align="center">block</td>
      <td align="center">scribble</td>
      <td align="center">block</td>
      <td align="center">point</td>
    </tr>
    <tr>
      <td align="center"><i>Origin</i></td>
      <td align="center"><a href="https://drive.google.com/drive/folders/10lZwgMFT6gV-Tj6gpFfd-TVJC3b9_808?usp=drive_link">Download</a></td> 
      <td align="center"><a href="https://drive.google.com/drive/folders/1HtHCYwpkqcuy5_BAt4gCiC5EN8ZUwMMW?usp=drive_link">Download</a></td>  
      <td align="center"><a href="https://drive.google.com/drive/folders/1xB3X0oM3fg5QIZ86LQTpHjlgmHURREV3?usp=drive_link">Download</a></td>  
      <td align="center"><a href="https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC">Download</a></td> 
      <td align="center"><a href="https://www.kaggle.com/datasets/debobratachakraborty/brats2019-dataset">Download</a></td>  
      <td align="center"><a href="https://ieeexplore.ieee.org/abstract/document/7872382">Paper</a></td>  
    </tr>
    <tr>
      <td align="center"><i>Processed (.h5)</i></td>
      <td align="center"><a href="https://github.com/LyxDLiI/ProCNS/tree/main/data">Download</a></td> 
      <td align="center"><a href="https://github.com/LyxDLiI/ProCNS/tree/main/data">Download</a></td>  
      <td align="center"><a href="https//:github.com/LyxDLiI/ProCNS/tree/main/data">Download</a></td>  
      <td align="center"><a href="https//:github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC">Download</a></td> 
      <td align="center"><a>N/A</a></td>  
      <td align="center"><a>N/A</a></td>  
    </tr>

  </tbody>
</table>

# Requirements
Some important required packages are listed below:
* Pytorch 1.10.2
* cudatoolkit 11.3.1
* efficientnet-pytorch 0.7.1
* tensorboardx 2.5.1
* medpy 0.4.0
* scikit-image 0.19.3
* simpleitk  2.1.1.2
* Python >= 3.9
# Usage
## 1. Clone this project
``` bash
git clone https://github.com/LyxDLiI/ProCNS.git
cd ProCNS/code
```

## 2. Create a conda environment
``` bash
conda env create -n procns -f procns.yaml
conda activate procns
pip install tree_filter-0.1-cp39-cp39-linux_x86_64.whl
```
## 3. Pre-processing
Data preprocessing includes normalizing all image intensities to between 0 and 1, while data augmentation includes randomly flipping images horizontally and vertically as well as rotation (spanning from -45° to 45°).

## 4. Train the model
``` bash 
bash run.sh
```

## 5. Test the model
``` bash
bash test.sh
```
## 6. Result
<div style="text-align: center;">
  <img src="image/output_table.png" alt="TEL">
</div>

## 7. Visualization
<div>
  <img src="image/output.png" alt="TEL">
</div>

# Citation
If you find FedLPPA useful in your research, please consider citing:
```
@article{liu2024procns,
  title={ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation},
  author={Liu, Y and Lin, L and Wong, KKY and Tang, X},
  journal={arXiv preprint arXiv:2401.14074},
  year={2024}
}
```
If you have any questions, please feel free to contact us.

# Acknowledgement
* [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)
* [FedICRA](https://github.com/llmir/FedICRA)
* [FedLPPA](https://github.com/llmir/FedLPPA)


