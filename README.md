
---
# Research-Deepfake

A lightweight toolkit for experimenting with deepfake image datasets, feature extraction, and exploratory analysis.

---

## Dataset Preparation

Download the following datasets and unzip them into `./data/raw`:

- **Fake images:** [1-million-fake-faces](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces)
- **Real images:** [flickrfaceshq-dataset-ffhq](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)

Your directory structure should look like:
data/
└── raw/
├── 1-million-fake-faces/
└── flickrfaceshq-dataset-ffhq/

## Source Code Overview

- `face_resize.py` — Resize raw images to 256×256 pixels.
- `face_crop.py` — Detect and crop face regions from raw images.
- `feature.py` — Extract FD, MFS, LAC, entropy, mean, and standard deviation features.
- `analyse.py` — Sample feature sets for distribution analysis.

---

## Quick Demo

```bash
cd ./scripts
sh pipeline.sh
