# MonoDF

MonoDF: Monocular 3D Object Detection with Depth Foundation Models

## Installation

1. Clone and create a conda environment:

```bash
git clone https://github.com/lixiaole2016/MonoDF-OPEN
cd MonoDF-OPEN

conda create -n monodf python=3.8
conda activate monodf
```

2. Install PyTorch matching your CUDA version (tested with torch 1.9+ / CUDA 11.x).

3. Install dependencies and compile deformable attention:

```bash
pip install -r requirements.txt

cd lib/models/monodf/ops/
bash make.sh
cd ../../../..
```

4. Download [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) ViT-S weights and place them at:

```
checkpoints/depth_anything_v2_vits.pth
```

5. Prepare [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

```
data/KITTI/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
├── training/
│   ├── image_2/
│   ├── calib/
│   └── label_2/
└── testing/
    └── ...
```

Adjust `dataset/root_dir` in the config if your KITTI path differs.

## Training

The default config is `configs/monodf.yaml` (Stage H: 195-epoch joint DA + OGM + GQR with conservative scheduled caps, ViT-S):

```bash
python tools/train_monodf.py
# or
bash train_monodf.sh --gpu 0
```

Checkpoints and logs are saved under `outputs/monodf/`.

## Evaluation

```bash
bash test.sh configs/monodf.yaml
```

Set `tester/checkpoint` in the config to the epoch you want to evaluate.

## Acknowledgements

This project is not possible without the following codebases.

- [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR)
- [MonoDGP](https://github.com/PuFan-001/MonoDGP)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
