# BarbieGait: An Identity-Consistent Synthetic Human Dataset with Versatile Cloth-Changing for Gait Recognition (CVPR 2026)

<p align="center">
  <img src="assets/BarbieGait.png" alt="BarbieGait main figure" width="100%">
</p>

<h4 align="center">
    <a href="https://barbiegait.github.io/" target="_blank">[🌐Website]</a> /
    <a href="" target="_blank">[📜Paper]</a>
</h4>

## 📖 Introduction

This is the official implementation of our paper presented at CVPR 2026: We propose BarbieGait, a synthetic gait dataset where real-world subjects are uniquely mapped into a virtual engine to simulate extensive clothing changes while preserving their gait identity information. As a pioneering work, BarbieGait provides a controllable gait data generation method, enabling the production of large datasets to validate cross-clothing issues that are difficult to verify with real-world data.

## 📂 Data Preparation

### Data Directory

All datasets should be placed in:
```
your_path/BarbieGait_data/
```

### Download and Extract

1. Download data from Google Drive to `BarbieGait_data/` directory

2. Decompress:
```bash
cd your_path/BarbieGait_data/
tar -xvjf BarbieGait_predsil_pkl.tar.bz2
```

3. Create symlink to code directory:
```bash
cd your_path/BarbieGait_CVPR26_release/BarbieGait
ln -s your_path/BarbieGait_CVPR26_release/BarbieGait_data ./BarbieGait_data
```

## 📂 Data Preprocessing

This step reorganizes raw data by renaming folders according to clothing labels for easier further study.

### Folder Structure

```
BarbieGait_data/
├── BarbieGait_predsil_pkl/        # Original data (personID/clothID-seqID)
├── thick_label_by_nakeddiffnorm_eqchg/  # Clothing thickness labels
└── P2_BarbieGait_predsil_pkl/    # Output: reorganized data
    └── {subject_id}/
        └── {cloth_type}/
            └── {view_id}/
                `-- {view_id}.pkl
```

### Usage

```bash
cd BarbieGait/datasets
python create_symlnk.py
```

### Output Format

Original folder names like `cloth00-00` are reorganized to `thick{thick_value}-{seq_in_thick}-cloth00-00`, where:
- `thick_value`: Clothing thickness category (0-9)
- `seq_in_thick`: Sequence index within that thickness category

This reorganization groups sequences by clothing thickness, facilitating further cross-clothing research.

## ✅ TODO

- [ ] Release the paper link
- [ ] Release the BarbieGait dataset
- [ ] Release the GaitCLIF codebase
- [ ] Release pretrained models and configs
- [ ] Improve documentation and usage examples
