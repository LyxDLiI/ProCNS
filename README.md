
# 🔔News
- 2024-12-23, 🎉🎉 Our paper "[**ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation**](https://arxiv.org/abs/2401.14074)" has been accepted by **IEEE Journal of Biomedical and Health Informatics (JBHI)**.
  
# ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation
The official implementation of the paper: [**ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation**](https://arxiv.org/abs/2401.14074)
![TEL](image/framework.png)
**Abstract:** Weakly-supervised segmentation (WSS) has emerged as a solution to mitigate the conflict between annotation cost and model performance by adopting sparse annotation formats (e.g., point, scribble, block, etc.). Typical approaches attempt to exploit anatomy and topology priors to directly expand sparse annotations into pseudo-labels. However, due to a lack of attention to the ambiguous edges in medical images and insufficient exploration of sparse supervision, existing approaches tend to generate erroneous and overconfident pseudo proposals in noisy regions, leading to cumulative model error and performance degradation. In this work, we propose a novel WSS approach, named ProCNS, encompassing two synergistic modules devised with the principles of progressive prototype calibration and noise suppression. Specifically, we design a Prototype-based Regional Spatial Affinity (PRSA) loss to maximize the pair-wise affinities between spatial and semantic elements, providing our model of interest with more reliable guidance. The affinities are derived from the input images and the prototype-refined predictions. Meanwhile, we propose an Adaptive Noise Perception and Masking (ANPM) module to obtain more enriched and representative prototype representations, which adaptively identifies and masks noisy regions within the pseudo proposals, reducing potential erroneous interference during prototype computation. Furthermore, we generate specialized soft pseudo-labels for the noisy regions identified by ANPM, providing supplementary supervision. Extensive experiments on three medical image segmentation tasks involving different modalities demonstrate that the proposed framework significantly outperforms representative state-of-the-art methods.

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


