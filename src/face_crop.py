#!/usr/bin/env python
# coding: utf-8

import cv2,logging,os
import argparse,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore, expon, kstest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from mtcnn import MTCNN
from tqdm import tqdm
tqdm.pandas()
import glob
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
from face_tool import get_face_image
from pathlib import Path
data_path = str(Path(__file__).resolve().parent.parent)+ "/data/"

def get_files(directory):
    png_files = glob.glob(f"{directory}/*.png")
    png_files_recursive = glob.glob(f"{directory}/**/*.png", recursive=True)
    return png_files_recursive

def if_processed(folder,file_name):
    if os.path.exists(f"{folder}{file_name}.png"):
        return True
    return False

def process_one_batch(fake_file_list,real_file_list,indexes):
    fake_folder = f"./data/face/crop/fake/"
    real_folder = f"./data/face/crop/real/"
    os.makedirs(fake_folder, exist_ok=True)
    os.makedirs(real_folder, exist_ok=True)
    detector = MTCNN(device="cpu")
    
    for index in indexes:
        if len(fake_file_list) > index:
            fake_file = fake_file_list[index]
            file_name = fake_file.split("/")[-1].split(".")[0]
            if not if_processed(fake_folder,file_name):
                face = get_face_image(detector,fake_file)
                if face.any():
                    #print("save",f"{fake_folder}{file_name}.png")
                    cv2.imwrite(f"{fake_folder}{file_name}.png",face)

        if len(real_file_list) > index:
            real_file = real_file_list[index]
            file_name = real_file.split("/")[-1].split(".")[0]
            if not if_processed(real_folder,file_name):
                face = get_face_image(detector,real_file)
                if face.any():
                    #print("save",f"{real_folder}{file_name}.png")
                    cv2.imwrite(f"{real_folder}{file_name}.png",face)

def main(worker):

    fake_folder = f"./data/face/resize/256/fake/"
    real_folder = f"./data/face/resize/256/real/"

    print(fake_folder)
    print(real_folder)

    fake_file_list = get_files(fake_folder)
    fake_file_list.sort()
    real_file_list = get_files(real_folder)
    real_file_list.sort()
    if len(real_file_list) == 0 or len(fake_file_list) == 0:
        batch_count = max(len(real_file_list), len(fake_file_list))
    else:
        batch_count = min(len(real_file_list), len(fake_file_list))
    
    if batch_count > 10000:
        batch_count = 10000
    
    batch_size = 50
    indexes_list = [list(range(start, min(start + batch_size, batch_count))) for start in range(0, batch_count, batch_size)]

    max_workers = worker
    #with ThreadPoolExecutor(max_workers=max_workers) as executor:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_batch, fake_file_list, real_file_list, batch) for batch in indexes_list]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch fractal or multifractal calculate.")
    parser.add_argument("--worker", type=int, default=8, help="int 0 to 64.")
    args = parser.parse_args()
    
    print(f"Worker selected : {args.worker}")

    main(args.worker)
