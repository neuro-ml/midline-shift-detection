# Incorporating Task-Specific Structural Knowledge into CNNs for Brain Midline Shift Detection

[[paper](https://arxiv.org/abs/1908.04568)]

This repository contains full code for training and inference of the model described in our 
[paper](https://arxiv.org/abs/1908.04568). 

## Install

```bash
git clone https://github.com/neuro-ml/midline-shift-detection
cd midline-shift-detection
pip install -r requirements.txt
pip install -e .
```

## Inference

```bash
python scripts/predict.py IMAGE OUTPUT_CONTOURS
# or run 
python scripts/predict.py --help
# for more details
```

All the images must be `.nii` or `.nii.gz` files containing axial MRI series.

## Training

```bash
python scripts/train.py DATA OUTPUT_MODEL
# or run
python scripts/train.py --help
# for more details
```

`DATA` is a folder containing the training set with the following structure:

```
DATA:
    - filename1.nii.gz
    - filename1.json
    - filename2.nii.gz
    - filename2.json
    ...
```

where the `json` files contain a list of annotations for a given image. See the [data](data) folder for an example.
