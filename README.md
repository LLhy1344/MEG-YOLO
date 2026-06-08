
````markdown
# MEG-YOLO: Boundary- and Context-Aware Detection of Low-Contrast Adrenal Structures in Non-Contrast CT

This repository provides the implementation of "MEG-YOLO", a boundary- and context-aware detection framework for low-contrast adrenal structure detection on routine non-contrast CT images.

This code repository is directly related to the manuscript currently under review:

> Title:"Boundary- and Context-Aware Detection of Low-Contrast Adrenal Structures in Non-Contrast CT for Primary Aldosteronism Imaging Review" 
> Authors:Hongyu Lv, Teng Ran, Guoliang Wang, Nanfang Li, Liang Yuan  
> Journal: "Under Review"  
> Year: 2026

MEG-YOLO is designed for "PA-related adrenal CT image review support". The model performs slice-level localization of the left and right adrenal glands and provides side-specific normal/abnormal reference-status prompts. It is not intended to replace adrenal venous sampling (AVS) or to perform direct patient-level functional lateralisation.

---

If you use this code in your research, please cite our paper:

```bibtex
@article{lv2026megyolo,
  title={Boundary- and Context-Aware Detection of Low-Contrast Adrenal Structures in Non-Contrast CT for Primary Aldosteronism Imaging Review},
  author={Lv, Hongyu and Ran, Teng and Wang, Guoliang and Li, Nanfang and Yuan, Liang},
  journal={Under Review},
  year={2026}
}
````


 Code and Data Availability

Code: Available in this repository.
Data: The original clinical CT dataset cannot be publicly shared because it contains private medical imaging data.
Benchmark protocol: The dataset construction, annotation rules, training/validation protocol, evaluation metrics, and key configuration files are provided to support reproducibility.
Model weights: Trained model weights can be released when permitted by the manuscript review and institutional policy.

Note: The model was trained using only CT images as input. AVS and clinical information were used only for reference label construction and were not provided to the model during training or inference.

---

## Task Description

Primary aldosteronism requires reliable adrenal assessment because treatment selection depends on distinguishing unilateral from bilateral disease. However, AVS is invasive, and non-contrast CT remains difficult to interpret visually.

In this work, adrenal CT review is formulated as a **slice-level detection task**:

* Detect visible adrenal regions on non-contrast axial CT images.
* Assign side-specific reference-status prompts.
* Use four joint side-status categories:

| Class ID | Label    | Definition                   |
| -------- | -------- | ---------------------------- |
| 0        | LAG(nor) | Normal left adrenal gland    |
| 1        | LAG(abn) | Abnormal left adrenal gland  |
| 2        | RAG(nor) | Normal right adrenal gland   |
| 3        | RAG(abn) | Abnormal right adrenal gland |

The normal/abnormal status is not defined by CT morphology alone. For PA patients, the status labels were assigned by integrating CT morphological findings with AVS-referenced clinical information. For non-PA controls, both adrenal sides were labeled as normal.

---

## Method Overview

MEG-YOLO follows a lightweight one-stage detection paradigm and introduces two task-oriented modules for small, low-contrast adrenal structure detection.

### 1. Multi-Scale Edge Information Enhancement, MSEIE

MSEIE is embedded in the backbone to strengthen weak adrenal boundaries and fine structural cues. It includes:

* Original-resolution local branch with 3 × 3 convolution.
* Multi-scale adaptive average pooling branches with scale bins `{3, 6, 9, 12}`.
* 1 × 1 convolution for channel compression.
* 3 × 3 depthwise convolution for local spatial modelling.
* EdgeEnhancer based on 3 × 3 average-pooling subtraction and residual edge fusion.

This module improves boundary-aware representation for small adrenal structures with weak contrast and irregular morphology.

### 2. Multi-scale Attention-Guided Local–Global Fusion, MAG-LGF

MAG-LGF is placed at the deepest stage of the backbone to enhance context-aware anatomical representation. It includes:

* 1 × 1 input projection.
* Channel-spatial attention recalibration.
* Structure-aware Multi-scale Context Aggregation, SMCA.
* Sequential max-pooling.
* Depthwise convolution with 5 × 5 and 9 × 9 kernels.
* Local-global fusion branch with 9 × 9 large-kernel depthwise convolution, multi-head self-attention, and convolutional FFN.

This module helps distinguish adrenal regions from adjacent vessels, renal margins, fat interfaces, liver, spleen, and other abdominal tissues with similar CT intensity.

---

## Availability of data and materials
The datasets used or analyzed during the current study are available from the corresponding author on reasonable request.






