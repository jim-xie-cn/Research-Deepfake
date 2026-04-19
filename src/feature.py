#!/usr/bin/env python
# coding: utf-8

import cv2,logging,os
import argparse,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImageLacunarity import CFAImageLacunarity
from FreeAeonFractal.FAImageFourier import CFAImageFourier
from FreeAeonFractal.FA2DMFS import CFA2DMFS
from FreeAeonFractal.FAImage import CFAImage
import seaborn as sns
import numpy as np
from scipy.stats import zscore, expon, kstest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.stats import skew, kurtosis
from mtcnn import MTCNN
from tqdm import tqdm
tqdm.pandas()
import glob
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
from extract_feature import CFeatureMatrix,get_lacunarity,get_mfs,get_fd,get_feature,get_img_feature,read_img,get_entropy

def get_files(directory):
    png_files = glob.glob(f"{directory}/*.png")
    png_files_recursive = glob.glob(f"{directory}/**/*.png", recursive=True)
    return png_files_recursive

def if_processed(folder,file_name):
    if os.path.exists(f"{folder}{file_name}.png"):
        return True
    return False

def get_stats(pca,img,file,folder):
    file_name = file.split("/")[-1].split(".")[0]
    result = []
    tmp = {}
    tmp['pca'] = pca
    pixels = img.flatten()
    tmp['entropy'] = get_entropy(img)
    tmp['mean'] = np.mean(pixels)
    tmp['std'] = np.std(pixels)
    tmp['skew'] = skew(pixels)
    tmp['kurt'] = kurtosis(pixels, fisher=True)  # fisher=True 返回过度峰度
    tmp['fd'] = get_fd(img, is_bin = False).to_dict(orient='records')
    
    tmp['common'] =  f"{folder}/{pca}-{file_name}-common.csv"
    tmp['mfs'] = f"{folder}/{pca}-{file_name}-mfs.csv"
    tmp['lac'] = f"{folder}/{pca}-{file_name}-lac.csv"
    
    df_common = pd.DataFrame([tmp])
    df_common.to_csv(tmp['common'])

    df_mfs = get_mfs(img, size = 128, is_bin = False)
    df_mfs.to_csv(tmp['mfs'])
    
    df_lac = get_lacunarity(img,size = 128, is_bin = False)
    df_lac.to_csv(tmp['lac'])

def get_data(file,folder):
    face = cv2.imread(file)
    if not face.any():
        return 
    
    file_name = file.split("/")[-1].split(".")[0]
    base_img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    get_stats(0,base_img,file,folder)
    
    for i in tqdm(range(1,130,1),desc=f"PCA feature {file_name}"):
        lowDData, reconData = CFAFeatureSelect.get_matrix_by_pca(base_img,i)
        delt = np.array(base_img - reconData)
        get_stats(i,delt,file,folder)
        #cv2.imwrite(f"{folder}/{i}-{file_name}.png",reconData)

    cv2.imwrite(f"{folder}/{file_name}.png",face)

def process_one_batch(params,indexes):
    action = params['action']
    fake_file_list = params['fake_file_list']
    real_file_list = params['real_file_list']
    fake_folder = params['output_fake_folder']
    real_folder = params['output_real_folder']
    
    try:
        for index in indexes:
            if len(fake_file_list) > index:
                fake_file = fake_file_list[index]
                file_name = fake_file.split("/")[-1].split(".")[0]
                if not if_processed(fake_folder,file_name):
                    df_tmp = get_data(fake_file,fake_folder)

            if len(real_file_list) > index:
                real_file = real_file_list[index]
                file_name = real_file.split("/")[-1].split(".")[0]
                if not if_processed(real_folder,file_name):
                    get_data(real_file,real_folder)
    
    except Exception:
        logging.exception("failed %d", index)
    
def test(fake_file_list,real_file_list):
    process_one_batch("train",fake_file_list,real_file_list,[0,1])

def main(dataset,worker,action):
    fake_folder = f"./data/face/crop/fake/"
    real_folder = f"./data/face/crop/real/"
    
    fake_file_list = get_files(fake_folder)
    fake_file_list.sort()
    real_file_list = get_files(real_folder)
    real_file_list.sort()

    if len(real_file_list) == 0 or len(fake_file_list) == 0:
        batch_count = max(len(real_file_list), len(fake_file_list))
    else:
        batch_count = min(len(real_file_list), len(fake_file_list))
    
    if batch_count > 50000:
        batch_count = 10000

    batch_size = 10
    
    params = {}
    params['dataset'] = dataset
    params['action'] = action
    params['fake_file_list'] = fake_file_list
    params['real_file_list'] = real_file_list
    params['output_fake_folder'] = f"./data/face/features/{dataset}/fake/"
    params['output_real_folder'] = f"./data/face/features/{dataset}/real/"
    
    os.makedirs(params['output_fake_folder'], exist_ok=True)
    os.makedirs(params['output_real_folder'], exist_ok=True)

    indexes_list = [list(range(start, min(start + batch_size, batch_count))) for start in range(0, batch_count, batch_size)]

    max_workers = worker
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_batch, params, batch) for batch in indexes_list]
        for future in tqdm(as_completed(futures), total=len(futures),desc=f"{dataset} feature"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch fractal or multifractal calculate.")
    parser.add_argument("--dataset", type=str, default='train', help="root path")
    parser.add_argument("--worker", type=int, default=8, help="int 0 to 64.")
    parser.add_argument("--action", choices=['common','mfs','lac'], default='common', help="action common/mfs/lac")
    args = parser.parse_args()
    
    print(f"Root selected: {args.dataset}")
    print(f"Worker selected : {args.worker}")
    print(f"Worker selected : {args.action}")

    main(args.dataset.strip(),args.worker,args.action.strip())

