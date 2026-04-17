---
# Research-Deepfake

A lightweight toolkit for experimenting with deepfake image datasets, feature extraction, and exploratory analysis.

---

## Dataset Preparation

Download the following datasets and unzip them into `./data/raw`:

- **Fake images:** [1-million-fake-faces](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces)
- **Real images:** [flickrfaceshq-dataset-ffhq](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)

Create the directory structure (if needed):

```bash
mkdir -p data/raw/{1-million-fake-faces,flickrfaceshq-dataset-ffhq}
```

After downloading and unzipping, the folders should look like:

```text
data/
└── raw/
    ├── 1-million-fake-faces/
    └── flickrfaceshq-dataset-ffhq/
```

---

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
```

This script runs the full preprocessing and analysis pipeline end-to-end.

---

## Notes

- Ensure all dependencies required by the scripts are installed (see `requirements.txt` if available).
- Adjust script paths or parameters as needed for your environment.

---