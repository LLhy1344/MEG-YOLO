
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






