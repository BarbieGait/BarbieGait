# BarbieGait Data Preparation

## Dataset Access

The BarbieGait dataset is hosted on [Hugging Face](https://huggingface.co/datasets/Andyen512/BarbieGait). Please fill out the access request manually. Requests are reviewed within one week. In case you encounter any issues, please contact [caiqingyuan@mail.bnu.edu.cn](mailto:caiqingyuan@mail.bnu.edu.cn).

## Download and Layout

Place dataset archives and extracted data under `your_path/BarbieGait_data/`.

```bash
cd your_path/BarbieGait_data
tar -xvjf BarbieGait_predsil_pkl.tar.bz2

cd your_path/BarbieGait_CVPR26_release/BarbieGait
ln -s /path/to/BarbieGait_data ./BarbieGait_data
```

The preprocessing scripts use the following layout:

```
BarbieGait_data/
└── BarbieGait_pred{modality}_pkl/
    └── {modality}_pkl/
        └── {subject_id}/cloth{cloth_id}-{sequence_id}/Camera{view_id}/Camera{view_id}.pkl
```

The clothing-thickness labels are provided in `datasets/BarbieGait/thick_label_by_nakeddiffnorm_eqchg/`.

## P2 Symbolic Links

`create_symlnk.py` builds a P2 view grouped by clothing-thickness labels. It creates symbolic links only and does not duplicate PKL data.

```bash
cd your_path/BarbieGait_CVPR26_release/BarbieGait
python datasets/create_symlnk.py --modality sil
```

Supported modalities are `sil`, `pose`, and `heatmap`. Use `--data-root` when `BarbieGait_data` is stored outside the repository:

```bash
python datasets/create_symlnk.py \
  --data-root /path/to/BarbieGait_data \
  --modality sil
```

The generated P2 layout is:

```
BarbieGait_pred{modality}_pkl/
├── {modality}_pkl/
│   └── {subject_id}/cloth{cloth_id}-{sequence_id}/
└── P2_pkl/
    └── {subject_id}/
        └── thick{thickness_label}-{index}-cloth{cloth_id}-{sequence_id}/
            └── Camera{view_id}/Camera{view_id}.pkl
```

`thickness_label` ranges from 0 to 9. `index` is the sequence index within that thickness category for a subject.

## Pose Heatmaps for GaitCLIF

GaitCLIF consumes two-channel pose heatmaps rather than raw COCO-17 pose coordinates. Generate heatmaps from `BarbieGait_predpose_pkl/pose_pkl`, then create their P2 view:

```bash
torchrun --standalone --nproc_per_node=8 datasets/pretreatment_heatmap.py \
  --pose_data_path /path/to/BarbieGait_data/BarbieGait_predpose_pkl/pose_pkl \
  --save_root /path/to/BarbieGait_data/BarbieGait_predheatmap_pkl/heatmap_pkl

python datasets/create_symlnk.py \
  --data-root /path/to/BarbieGait_data \
  --modality heatmap
```

Set `--nproc_per_node` to the number of available GPUs.
