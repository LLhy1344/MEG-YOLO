
````markdown
# MEG-YOLO: Boundary- and Context-Aware Detection of Low-Contrast Adrenal Structures in Non-Contrast CT

This repository provides the implementation of **MEG-YOLO**, a boundary- and context-aware detection framework for low-contrast adrenal structure detection on routine non-contrast CT images.

The code is associated with the manuscript:

> **Boundary- and Context-Aware Detection of Low-Contrast Adrenal Structures in Non-Contrast CT for Primary Aldosteronism Imaging Review**  
> Hongyu Lv, Teng Ran, Guoliang Wang, Nanfang Li, Liang Yuan  
> Journal: *Under Review*  
> Year: 2026

MEG-YOLO is designed for **PA-related adrenal CT image review support**. The model performs slice-level localization of the left and right adrenal glands and provides side-specific normal/abnormal reference-status prompts. It is not intended to replace adrenal venous sampling (AVS) or to perform direct patient-level functional lateralisation.

---

## Related Manuscript

This code repository is directly related to the manuscript currently under review:

```bibtex
@article{lv2026megyolo,
  title={Boundary- and Context-Aware Detection of Low-Contrast Adrenal Structures in Non-Contrast CT for Primary Aldosteronism Imaging Review},
  author={Lv, Hongyu and Ran, Teng and Wang, Guoliang and Li, Nanfang and Yuan, Liang},
  journal={Under Review},
  year={2026}
}
````

Please consider citing our work if you find this repository useful.

---

## Code and Data Availability

* **Code:** Available in this repository.
* **Data:** The original clinical CT dataset cannot be publicly shared because it contains private medical imaging data.
* **Benchmark protocol:** The dataset construction, annotation rules, training/validation protocol, evaluation metrics, and key configuration files are provided to support reproducibility.
* **Model weights:** Trained model weights can be released when permitted by the manuscript review and institutional policy.

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

## Dataset

The internal dataset was retrospectively collected from the People's Hospital of Xinjiang Uygur Autonomous Region between January 2024 and December 2024.

### Image-level dataset

| Item                      | Training set | Validation set | Total |
| ------------------------- | -----------: | -------------: | ----: |
| CT images                 |         1444 |            361 |  1805 |
| LAG(abn)                  |          652 |            156 |   808 |
| LAG(nor)                  |          607 |            149 |   756 |
| RAG(abn)                  |          385 |             94 |   479 |
| RAG(nor)                  |          717 |            178 |   895 |
| Total annotated instances |         2361 |            577 |  2938 |

The image-level dataset was randomly divided into training and validation sets at a ratio of 8:2. The validation set was used only for internal performance evaluation and was not used for early stopping or iterative hyperparameter tuning.

### Additional patient-level validation cohort

An additional patient-level validation cohort was used to provide a more conservative estimate of patient-independent performance.

| Item                         | Value |
| ---------------------------- | ----: |
| Patients with bilateral PA   |    10 |
| Patients with left-sided PA  |     8 |
| Patients with right-sided PA |     7 |
| Non-PA controls              |     5 |
| CT images                    |   352 |
| Annotated adrenal instances  |   507 |

This cohort was used only for patient-level validation with fixed trained weights. No additional training, fine-tuning, or threshold adjustment was performed.

---

## Installation

This project is based on PyTorch and the Ultralytics YOLO framework.

```bash
# Create environment
conda create -n megyolo python=3.10
conda activate megyolo

# Install PyTorch according to your CUDA version
pip install torch torchvision torchaudio

# Install Ultralytics
pip install ultralytics

# Install other dependencies
pip install -r requirements.txt
```

The experimental environment used in the manuscript was:

| Component | Version / Configuration |
| --------- | ----------------------- |
| Python    | 3.10.16                 |
| PyTorch   | 2.6.0                   |
| CUDA      | 12.0                    |
| GPU       | NVIDIA RTX A5000, 24 GB |
| CPU       | Intel Core i9-12900KF   |

---

## Repository Structure

A recommended repository structure is shown below:

```text
MEG-YOLO/
├── README.md
├── requirements.txt
├── ultralytics/
│   ├── cfg/
│   │   ├── models/
│   │   │   └── v8/
│   │   │       └── MESIS-YOLO-2.yaml
│   │   └── datasets/
│   │       └── adrenal_ct.yaml
│   ├── nn/
│   └── engine/
├── datasets/
│   └── adrenal_ct/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── runs/
│   └── detect/
├── weights/
│   └── best.pt
└── scripts/
    ├── train_megyolo.py
    ├── val_megyolo.py
    └── predict_megyolo.py
```

The exact file paths can be modified according to your local environment.

---

## Dataset Format

The dataset follows the standard YOLO detection format.

Each image has a corresponding `.txt` label file:

```text
class_id x_center y_center width height
```

where the bounding-box coordinates are normalized to `[0, 1]`.

Example:

```text
0 0.5123 0.4387 0.0831 0.0524
2 0.4215 0.4462 0.0758 0.0479
```

The dataset YAML file can be written as:

```yaml
path: /path/to/adrenal_ct
train: images/train
val: images/val

names:
  0: LAG(nor)
  1: LAG(abn)
  2: RAG(nor)
  3: RAG(abn)
```

---

## Training

The main training configuration used in the manuscript is:

```yaml
task: detect
mode: train
model: ultralytics/cfg/models/v8/MESIS-YOLO-2.yaml
data: 3.yaml
epochs: 300
batch: 4
imgsz: 1280
device: '0'
workers: 0
pretrained: true
optimizer: auto
seed: 0
deterministic: true
amp: true
patience: 0
save: true
save_period: 1

lr0: 0.005
lrf: 0.001
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0

box: 7.5
cls: 0.5
dfl: 1.5

translate: 0.1
scale: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.1
copy_paste_mode: flip
fliplr: 0.5
flipud: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
auto_augment: randaugment
erasing: 0.4
close_mosaic: 20
```

Train MEG-YOLO using the Ultralytics CLI:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/MESIS-YOLO-2.yaml \
  data=3.yaml \
  epochs=300 \
  batch=4 \
  imgsz=1280 \
  device=0 \
  workers=0 \
  seed=0 \
  deterministic=True \
  pretrained=True \
  lr0=0.005 \
  lrf=0.001 \
  weight_decay=0.0005 \
  save=True \
  save_period=1
```

Alternatively, train using Python:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/MESIS-YOLO-2.yaml")

results = model.train(
    data="3.yaml",
    epochs=300,
    batch=4,
    imgsz=1280,
    device=0,
    workers=0,
    seed=0,
    deterministic=True,
    pretrained=True,
    optimizer="auto",
    lr0=0.005,
    lrf=0.001,
    weight_decay=0.0005,
    save=True,
    save_period=1,
)
```

Training outputs are saved in:

```text
runs/detect/train*/
```

---

## Validation and Evaluation

During validation and evaluation, the following post-processing settings were used:

| Setting                    | Value    |
| -------------------------- | -------- |
| Confidence threshold       | 0.001    |
| NMS IoU threshold          | 0.7      |
| Maximum detections         | 300      |
| Test-time augmentation     | Disabled |
| Additional post-processing | Not used |

Run validation:

```bash
yolo detect val \
  model=runs/detect/train12/weights/best.pt \
  data=3.yaml \
  imgsz=1280 \
  batch=4 \
  device=0 \
  conf=0.001 \
  iou=0.7
```

Python validation:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train12/weights/best.pt")

metrics = model.val(
    data="3.yaml",
    imgsz=1280,
    batch=4,
    device=0,
    conf=0.001,
    iou=0.7,
)
```

The following metrics are reported:

* Precision
* Recall
* F1-score
* mAP50
* mAP50–95
* AP for small and medium objects under COCO area definitions
* Params
* GFLOPs
* Average per-image inference time

---

## Main Results

### Internal image-level validation

| Model     | Type              | Precision | Recall | mAP50 | mAP50–95 | Params (M) | GFLOPs |
| --------- | ----------------- | --------: | -----: | ----: | -------: | ---------: | -----: |
| YOLOv8    | One-stage CNN     |     0.858 |  0.859 | 0.924 |    0.581 |        3.0 |    8.1 |
| YOLOv10   | One-stage CNN     |     0.823 |  0.821 | 0.892 |    0.557 |        2.7 |    8.2 |
| YOLOv11   | One-stage CNN     |     0.835 |  0.875 | 0.918 |    0.567 |        2.6 |    6.3 |
| YOLOv12   | One-stage CNN     |     0.828 |  0.888 | 0.910 |    0.562 |        2.5 |    5.8 |
| YOLO-TLP  | One-stage CNN     |     0.833 |  0.799 | 0.888 |    0.547 |        1.9 |    9.5 |
| DEIMv2    | Transformer-based |     0.792 |  0.642 | 0.904 |    0.521 |      18.05 |  51.86 |
| RT-DETRv4 | Transformer-based |     0.815 |  0.656 | 0.918 |    0.543 |      19.39 |  56.35 |
| MEG-YOLO  | One-stage CNN     |     0.898 |  0.882 | 0.946 |    0.598 |        3.2 |    7.9 |

### Additional patient-level validation

| Model    | Precision | Recall |    F1 | mAP50 | mAP50–95 |
| -------- | --------: | -----: | ----: | ----: | -------: |
| YOLOv8   |     0.664 |  0.775 | 0.715 | 0.723 |    0.371 |
| MEG-YOLO |     0.698 |  0.780 | 0.737 | 0.734 |    0.380 |

The patient-level validation results provide a more conservative estimate of patient-independent performance. MEG-YOLO retained a modest advantage over YOLOv8, but reliable patient-independent adrenal CT detection remains challenging because of small adrenal size, weak boundary contrast, close anatomical neighbourhoods, and normal-abnormal status ambiguity.

---

## Ablation Study

### MSEIE ablation

| Model configuration                | Precision | Recall |    F1 | mAP50 | mAP50–95 |
| ---------------------------------- | --------: | -----: | ----: | ----: | -------: |
| YOLOv8                             |     0.858 |  0.859 | 0.858 | 0.924 |    0.581 |
| YOLOv8/MSEIE                       |     0.873 |  0.889 | 0.881 | 0.933 |    0.589 |
| YOLOv8/MSEIE w/o EdgeEnhancer      |     0.866 |  0.875 | 0.870 | 0.927 |    0.584 |
| YOLOv8/MSEIE with single-scale bin |     0.868 |  0.878 | 0.873 | 0.930 |    0.589 |
| MEG-YOLO                           |     0.898 |  0.882 | 0.889 | 0.946 |    0.598 |

### MAG-LGF ablation

| Model configuration              | Precision | Recall |    F1 | mAP50 | mAP50–95 |
| -------------------------------- | --------: | -----: | ----: | ----: | -------: |
| YOLOv8                           |     0.858 |  0.859 | 0.858 | 0.924 |    0.581 |
| YOLOv8/MAG-LGF                   |     0.869 |  0.886 | 0.877 | 0.940 |    0.585 |
| MEG-YOLO w/o Local-Global Fusion |     0.894 |  0.880 | 0.886 | 0.920 |    0.578 |
| MEG-YOLO w/o SMCA branch         |     0.853 |  0.895 | 0.873 | 0.931 |    0.590 |
| MEG-YOLO                         |     0.898 |  0.882 | 0.889 | 0.946 |    0.598 |

The ablation results indicate that MSEIE improves boundary-aware representation and that MAG-LGF contributes additional context-aware discrimination.

---

## Prediction

Run prediction on a single image:

```bash
yolo detect predict \
  model=runs/detect/train12/weights/best.pt \
  source=/path/to/ct_image.jpg \
  imgsz=1280 \
  conf=0.25 \
  iou=0.7 \
  save=True
```

Run prediction on a folder:

```bash
yolo detect predict \
  model=runs/detect/train12/weights/best.pt \
  source=/path/to/images \
  imgsz=1280 \
  conf=0.25 \
  iou=0.7 \
  save=True
```

Note: For formal validation and mAP calculation, use `conf=0.001`. For visualization, a higher confidence threshold such as `0.25` may be used for clearer display.

---

## Reproducibility Statement

To improve reproducibility, all experiments were conducted using a fixed random seed of 0 and deterministic training enabled. All CT images were resized to 1280 × 1280 and normalized before training. Data augmentation was applied only to the training set, including random scaling, translation, mosaic augmentation, mild intensity perturbation, and horizontal flipping with anatomical label remapping. No random augmentation was used during validation.

All comparison methods used the same training and validation split, input resolution, preprocessing procedure, augmentation strategy, optimizer, learning-rate setting, batch size, epoch number, and evaluation metrics. The same checkpoint-selection rule was applied to all models, and the best-performing validation checkpoint under the unified protocol was used for final evaluation.

Because the clinical CT dataset cannot be publicly shared, this repository provides the implementation, configuration files, benchmark protocol, and evaluation scripts to support reproducible research.

---

## Important Notes

1. MEG-YOLO is intended for computer-assisted adrenal CT image review.
2. The model is not designed to replace AVS-based functional lateralisation.
3. The normal/abnormal labels are side-specific reference-status labels derived from CT morphology and clinical reference information.
4. The current framework is based on 2D CT slices and does not fully exploit 3D adrenal morphology or inter-slice continuity.
5. Larger patient-level cohorts, external multicentre validation, volumetric modelling, and integration with biochemical variables are still required before clinical application.

---

## Contact

For questions about the code or manuscript, please contact:

* Hongyu Lv: [1344108528@qq.com](mailto:1344108528@qq.com)
* Teng Ran: [ranteng@xju.edu.cn](mailto:ranteng@xju.edu.cn)
* Nanfang Li: [lanfang2016@sina.com](mailto:lanfang2016@sina.com)
* Liang Yuan: [lyuan@sjtu.edu.cn](mailto:lyuan@sjtu.edu.cn)

---

## License

This repository is released for academic research purposes. Please check the license file for details.

---

## Acknowledgement

This implementation is based on the Ultralytics YOLO framework. We thank the clinicians and radiologists involved in CT image review, annotation, and clinical reference confirmation.

```
```

[1]: https://github.com/fciasth/EMO-YOLO/tree/main "GitHub - fciasth/EMO-YOLO · GitHub"
