# TXL-PBC: A Curated and Re-annotated Peripheral Blood Cell Dataset

## Overview

TXL-PBC is a curated and re-annotated peripheral blood cell dataset constructed by integrating four publicly available resources:  
- Blood Cell Count and Detection (BCCD)  
- Blood Cell Detection Dataset (BCDD)  
- Peripheral Blood Cells (PBC)  
- Raabin White Blood Cell (Raabin-WBC)  

The dataset contains 1,260 images and 18,143 bounding box annotations for three major blood cell types:  
- White blood cells (WBC)  
- Red blood cells (RBC)  
- Platelets (PC)  

All images are annotated in YOLO format and split into training, validation, and test sets.

## Directory Structure

```
TXL-PBC/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
├── BCCD_selection.xlsx
├── classes.txt
├── metadata_file.xlsx
└── annotation_protocol.pdf
```

- `images/`: Contains all images, split into train/val/test.
- `labels/`: YOLO-format annotation files, split into train/val/test.
- `data.yaml`: YOLO configuration file.
- `BCCD_selection.xlsx`: List of selected and excluded BCCD images.
- `metadata.csv`: Mapping of image filenames to source datasets.
- `annotation_protocol.pdf`: Manual annotation guidelines.

## Usage

- The dataset can be used directly with object detection frameworks such as YOLO.
- Recommended preprocessing: image normalization and data augmentation.
- Suitable for training, validation, and benchmarking of blood cell detection models.
- Can be combined with other datasets for cross-validation or transfer learning.

## YOLO26 Fine-Tuning

Install dependencies:

```bash
python3 -m pip install -U -r requirements.txt
```

Run preprocessing plus training with defaults:

```bash
scripts/train_yolo26.sh
```

Choose the YOLO26 model size and training hyperparameters from the terminal:

```bash
scripts/train_yolo26.sh --size s --epochs 150 --batch 8 --device 0 --lr0 0.005
```

The supported `--size` values are `n`, `s`, `m`, `l`, and `x`. The script prepares a cleaned dataset under `runs/datasets/txl_pbc_preprocessed/`, then fine-tunes `yolo26<SIZE>.pt` and writes training artifacts under `runs/yolo26/`.
If the selected YOLO26 weights are not available locally, the training wrapper downloads them from the latest Ultralytics assets release into `runs/weights/` before training starts.

Training saves `best.pt` and `last.pt` under `runs/yolo26/<run_name>/weights/` by default. To resume an interrupted run, point `--resume` at the previous `last.pt` checkpoint:

```bash
scripts/train_yolo26.sh --resume runs/yolo26/txl_pbc_yolo26s/weights/last.pt --device 0
```

To start from any explicit checkpoint or model file:

```bash
scripts/train_yolo26.sh --model path/to/checkpoint.pt --size s --device 0
```

Each completed run writes deployment metadata under `runs/yolo26/<run_name>/deployment_metadata/`, including `metadata.json`, `metadata.yaml`, a copied `data.yaml`, optional preprocessing report, checkpoint hashes, class names, training hyperparameters, package versions, CUDA information, and a short model card.

Training history is appended under `runs/yolo26/<run_name>/training_history/`:

```text
history.jsonl
history.csv
pretrain_metrics.json
run_config.json
```

Before epoch 1, the script runs a baseline validation by default and stores it as an `epoch: 0` / `event: pretrain_eval` row. For object detection, accuracy is represented by precision, recall, `mAP50`, and `mAP50-95`; loss values are included when Ultralytics exposes them during standalone validation. When resuming from `last.pt`, history continues in the same run directory.

You can change or disable the baseline split:

```bash
scripts/train_yolo26.sh --size m --pretrain-eval-split test
scripts/train_yolo26.sh --size m --no-pretrain-eval
```

You can run the preprocessing step directly:

```bash
python3 preprocess_yolo_dataset.py --source TXL-PBC --output runs/datasets/txl_pbc_preprocessed
```

Create annotated prediction samples from the test split:

```bash
scripts/test_yolo26_samples.sh --model runs/yolo26/txl_pbc_yolo26m2/weights/best.pt --device 0 --num-samples 20
```

Annotated images are saved as `*_pred.jpg` under `runs/prediction_samples/`. The script also writes `predictions.csv`, `predictions.json`, and `summary.json`. Add `--show-ground-truth` to overlay the original YOLO labels in gray for visual comparison.

For all training options:

```bash
scripts/train_yolo26.sh --help
```



## License
This dataset is licensed under the [MIT License](LICENSE).

## Citing the TXL-PBC Dataset

If you use this dataset in your research, please cite our paper:

> Gan, L., Li, X. & Wang, X. A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources. *Sci Data* **12**, 1694 (2025).  
> https://doi.org/10.1038/s41597-025-05980-z

**BibTeX entry:**
```bibtex
@article{gan2025curated,
  title={A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources},
  author={Gan, Lu and Li, Xi and Wang, Xichun},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1694},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
