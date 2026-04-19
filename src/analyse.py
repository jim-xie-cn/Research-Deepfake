import cv2,logging
import argparse
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier
from FreeAeonFractal.FAImage import CFAImage
#GPU version
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
from FreeAeonFractal.FAImageLacunarity import CFAImageLacunarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore, expon, kstest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from tqdm import tqdm
tqdm.pandas()
import glob,os
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from extract_feature import CFeatureMatrix,get_lacunarity,get_mfs,get_fd,get_feature,get_img_feature,read_img,get_fourier_image
from common import get_stats,show_stats,show_t_sne,show_pca,show_mfs

def get_files(directory):
    png_files = glob.glob(f"{directory}/*.csv")
    png_files_recursive = glob.glob(f"{directory}/**/*.csv", recursive=True)
    return png_files_recursive

def load_sample(dataset,action):
    fake_folder = get_files(f"./data/face/features/{dataset}/fake/")
    real_folder =  get_files(f"./data/face/features/{dataset}/real/")
    result = []
    for file in fake_folder:
        tmp = file.split(".csv")[0].split("/")[-1].split("-")
        pca = int(tmp[0])
        file_id = tmp[1]
        type = tmp[2]
        if type != action:
            continue
        df_tmp = pd.read_csv(file)
        df_tmp['kind'] = "fake"
        df_tmp['pca'] = pca
        df_tmp['file'] = file_id
        result.append(df_tmp)

    for file in real_folder:
        tmp = file.split(".csv")[0].split("/")[-1].split("-")
        pca = int(tmp[0])
        file_id = tmp[1]
        type = tmp[2]
        if type != action:
            continue
        df_tmp = pd.read_csv(file)
        df_tmp['kind'] = "real"
        df_tmp['pca'] = pca
        df_tmp['file'] = file_id
        result.append(df_tmp)

    return pd.concat(result,ignore_index=True).reset_index(drop = True)

def load_group(dataset,action):
    df = load_sample(dataset,action)
    if action == 'common':
        grouped_data = list(df.groupby(['kind', 'pca']))
    elif action == 'mfs':
        grouped_data = list(df.groupby(['kind', 'pca','q']))
    elif action == 'lac':
        grouped_data = list(df.groupby(['kind', 'pca','scales']))
    return grouped_data

def merge_group(item_df_tuple,action):
    item, df_t = item_df_tuple
    data_list = []
    for i in tqdm(range(10),desc=f"{item[0]}-{item[1]}"):
        df_sample = df_t.sample(n=500,replace=True)
        df_stats = get_stats(df_sample)
        df_stats['kind'] = item[0]
        df_stats['pca'] = item[1]
        if action == 'mfs':
            df_stats['q'] = item[2]
        elif action == 'lac':
            df_stats['scales'] = item[2]
        data_list.append(df_stats)
    
    return pd.concat(data_list, ignore_index=True).reset_index(drop = True)

def process_group(action,item_df_tuple):
    return merge_group(item_df_tuple,action)

def main(dataset,action,worker):
    grouped_data = load_group(dataset,action)
    results = []
    with ThreadPoolExecutor(max_workers=worker) as executor:
    #with ProcessPoolExecutor(max_workers=worker) as executor:  
        futures = [executor.submit(process_group,action, g) for g in grouped_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing groups {action}"):
            results.append(future.result())

    if results:
        df_stats = pd.concat(results, ignore_index=True).reset_index(drop=True)
        os.makedirs(f"./data/face/stats/{dataset}", exist_ok=True)
        df_stats.to_csv(f"./data/face/stats/{dataset}/{action}.csv")
    else:
        print("no result")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch fractal or multifractal calculate.")
    parser.add_argument("--dataset", type=str, default='train', help="root path")
    parser.add_argument("--action", choices=['common','mfs','lac'], default='common',help='actions')
    parser.add_argument("--worker", type=int, default=12, help="int 0 to 24.")
    args = parser.parse_args()
    print(args)
    main(args.dataset.strip(),args.action.strip(),args.worker)
