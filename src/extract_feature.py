import cv2
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from PIL import Image

from FreeAeonFractal.FAImageFourier import CFAImageFourier
from FreeAeonFractal.FAImage import CFAImage
from FreeAeonFractal.FAImageLacunarity import CFAImageLacunarity
#from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension
#from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FA2DMFS import CFA2DMFS
from FreeAeonFractal.FAImageLacunarity import CFAImageLacunarity
from FreeAeonML.FADataEDA import CFAFitter

fitter = CFAFitter(fitter = 'polynomial')
def get_fit_data(x,coefficients):
    return fitter.get_fit_data(x,coefficients)

def get_fit_params(x,y):
    return fitter.fit(x,y)

def get_entropy(mat):
    arr = np.ravel(mat)
    values, counts = np.unique(arr, return_counts=True)
    prob = counts / counts.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def get_pass_mask(shape, low_freq_r):
    h, w = shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    low_mask = dist < float(low_freq_r)
    high_mask = dist > float(low_freq_r)
    #return low_mask,high_mask
    tol = 1 
    return np.abs(dist - float(low_freq_r)) <= tol

def get_fourier_image(gray_image, size):
    fourier = CFAImageFourier(gray_image)
    mag, phase = fourier.get_raw_spectrum()
    #h, w = mag[0].shape
    if size == 32:
        rs = np.arange(2, 130, 4)
    elif size == 64:
        rs = np.arange(2, 130, 2)
    elif size == 128:
        rs = np.arange(2, 130, 1)
    elif size == 256:
        rs = np.arange(2, 130, 0.5)
    elif size == 512:
        rs = np.arange(2, 130, 0.25)
    result = []
    for r in rs:
        mask = get_pass_mask(gray_image.shape, r)
        mag_img = (np.array(mag[0])*mask)
        phase_img = (np.array(phase[0])*mask)
        mask_img = fourier.extract_by_freq_mask(mask)
        result.append((r,mask_img,mag_img,phase_img))
    #print("fourier count",len(result))
    return result

def get_lacunarity(img,size,is_bin):
    # 40 -- 32 97 -- 64, 265 -- 128
    if size == 32:
        max_scales = 64 #40
    elif size == 64:
        max_scales = 128 #97
    elif size == 128:
        max_scales = 256 #565
    else:
        raise FileNotFoundError(f"Unsupported size",size)
    max_size = min(img.shape)
    max_size = min(max_size,128)
    lac = CFAImageLacunarity(img,max_size=max_size,max_scales=max_scales,with_progress=False,scales_mode="logspace")
    #lac = CFAImageLacunarity(img,max_size=max_size,max_scales=max_scales,with_progress=False,scales_mode="powers")
    res = lac.get_lacunarity(corp_type=-1, use_binary_mass=is_bin, include_zero=True)
    df = pd.DataFrame()
    df['scales'] = pd.Series(res['scales'])
    df['lacunarity'] = pd.Series(res['lacunarity'])
    return df

def get_mfs(img,size,is_bin):
    MFS = CFA2DMFS(img,
                   with_progress = False,
                   q_list=np.linspace(-8, 8, size),
                   bg_reverse=False,
                   bg_threshold=0,
                   bg_otsu=is_bin,
                   mu_floor=1e-12)
    max_size = min(img.shape)
    max_scales = max_size // 2
    df_mass, df_fit, df_spec = MFS.get_mfs(max_size=max_size,
                                           max_scales=max_scales,
                                           min_points=5,
                                           min_box=2,
                                           use_middle_scales=False,
                                           if_auto_line_fit=False,
                                           fit_scale_frac=(0.3, 0.7),
                                           auto_fit_min_len_ratio=0.4,
                                           cap_d0_at_2=False)

    #df_mass, df_fit, df_spec = MFS.get_mfs(max_size=max_size,
    #                                       max_scales=max_scales,
    #                                       min_points=1,
    #                                       min_box=2,
    #                                       use_middle_scales=False,
    #                                       if_auto_line_fit=False,
    #                                       fit_scale_frac=(0.1, 0.9),
    #                                       auto_fit_min_len_ratio=0.1,
    #                                       cap_d0_at_2=False)

    df_spec = df_spec.rename(columns={'tau':'t(q)','Dq': 'd(q)', 'alpha': 'a(q)', 'f_alpha':'f(a)'})
    if not df_spec.empty:
        del df_spec['D1']
    #print("mfs count",df_spec['q'].nunique())
    return df_spec

def get_fd(img,is_bin = False):
    if is_bin:
        fd = CFAImageDimension(img,with_progress = False).get_bc_fd(corp_type=-1)
    else:
        fd = CFAImageDimension(img,with_progress = False).get_dbc_fd(corp_type=-1)
        #fd_sdbc = CFAImageDimension(gray_image,with_progress = False).get_sdbc_fd(corp_type=-1)
    if is_bin:
        return pd.DataFrame([{"bin":"yes","fd":fd['fd']}])
    else:
        return pd.DataFrame([{"bin":"no","fd":fd['fd']}])

def get_img_feature(img,size,action):
    result = []
    bin_img,threshold = CFAImage.otsu_binarize(img)
    if 'fd' == action:
        fd_bc = CFAImageDimension(bin_img,with_progress = False).get_bc_fd(corp_type=-1)
        df_tmp = pd.DataFrame([{"bin":"yes","fd":fd_bc['fd']}])
        result.append(df_tmp)

        fd_dbc = CFAImageDimension(img,with_progress = False).get_dbc_fd(corp_type=-1)
        df_tmp = pd.DataFrame([{"bin":"no","fd":fd_dbc['fd']}])
        result.append(df_tmp)
    
    elif 'lac' == action:
        df_tmp = get_lacunarity(bin_img, size = size,is_bin = True )
        df_tmp['bin'] = 'yes'
        result.append(df_tmp)
        
        df_tmp = get_lacunarity(img, size = size,is_bin = False )
        df_tmp['bin'] = 'no'
        result.append(df_tmp)
    
    elif 'mfs' == action:
        df_tmp = get_mfs(bin_img, size = size ,is_bin = True)
        df_tmp['bin'] = 'yes'
        result.append(df_tmp)
        
        df_tmp = get_mfs(img, size = size ,is_bin = False)
        df_tmp['bin'] = 'no'
        result.append(df_tmp)

    elif 'entropy' == action:
        entropy = get_entropy(bin_img)
        df_tmp = pd.DataFrame([{"bin":"yes","entropy":entropy}])
        result.append(df_tmp)
        
        entropy = get_entropy(img)
        df_tmp = pd.DataFrame([{"bin":"no","entropy":entropy}])
        result.append(df_tmp)

    return pd.concat(result,ignore_index=True).reset_index(drop = True)

def read_img(file_name):
    image = cv2.imread(file_name)
    img_norm = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)
    return img_norm

def get_feature(file_name, size = 32, with_img = False, actions = ['fd','lac','mfs','fourier','entropy']):
    image = read_img(file_name)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b_img, g_img, r_img = cv2.split(image)
    result = {
        "fd":[],
        "lac":[],
        "mfs":[],
        "fourier":[],
        "entropy":[]
    }
    for type,image in zip(['gray','b','g','r'],[gray_img,b_img,g_img,r_img]):
        imgs = get_fourier_image(image, size = size)
        for item in tqdm(imgs,desc=f"{file_name}:{type}"):
            r = item[0]
            mask_img = item[1]
            mag_img = item[2]
            phase_img = item[3]
            
            if 'fourier' in actions:
                tmp = {}
                tmp['r'] = r
                tmp['img'] = type
                tmp['magnitude_mean'] = mag_img.mean()
                tmp['magnitude_std'] = mag_img.std()
                tmp['phase_mean'] = phase_img.mean()
                tmp['phase_std'] = phase_img.std()
                if with_img:
                    tmp['image'] = mask_img
                result['fourier'].append(pd.DataFrame([tmp]))

            if 'entropy' in actions:
                tmp = {}
                tmp['magnitude'] = get_entropy(mag_img)
                tmp['phase'] = get_entropy(phase_img)
                tmp['mask'] = get_entropy(mask_img)
                tmp['r'] = r
                tmp['img'] = type
                result['entropy'].append(pd.DataFrame([tmp]))

            #FD
            if 'fd' in actions:
                df_tmp = get_img_feature(mask_img,size,'fd')
                df_tmp['r'] = r
                df_tmp['img'] = type
                result['fd'].append(df_tmp)

            # LAC
            if 'lac' in actions:
                df_tmp = get_img_feature(mask_img,size,'lac')
                df_tmp['img'] = type
                df_tmp['r'] = r
                result['lac'].append(df_tmp)
            
            #MFS
            if 'mfs' in actions:
                df_tmp = get_img_feature(mask_img,size,'mfs')
                df_tmp['img'] = type
                df_tmp['r'] = r
                result['mfs'].append(df_tmp)
    
    if result['fd']:
        df_fd = pd.concat(result['fd'],ignore_index = True).reset_index(drop = True)
    else:
        df_fd = pd.DataFrame()
    if result['lac']:
        df_lac = pd.concat(result['lac'],ignore_index = True).reset_index(drop = True)
    else:
        df_lac = pd.DataFrame()
    if result['mfs']:
        df_mfs = pd.concat(result['mfs'],ignore_index = True).reset_index(drop = True)
    else:
        df_mfs = pd.DataFrame()
    if result['fourier']:
        df_fourier = pd.concat(result['fourier'],ignore_index = True).reset_index(drop = True)
    else:
        df_fourier = pd.DataFrame()
    if result['entropy']:
        df_entropy = pd.concat(result['entropy'],ignore_index = True).reset_index(drop = True)
    else:
        df_entropy = pd.DataFrame()
    return df_fd,df_lac,df_mfs,df_fourier,df_entropy


class CFeatureMatrix:

    def __init__(self,df_fd,df_lac,df_mfs):
        self.m_fd = df_fd
        self.m_lac = df_lac
        self.m_mfs = df_mfs

    @staticmethod
    def extract(file,size = 32,with_img=False,actions=['fd','lac','mfs','fourier','entropy']):
        df_fd,df_lac,df_mfs,df_fourier,df_entropy = get_feature(file,size,with_img,actions=actions)
        return df_fd,df_lac,df_mfs,df_fourier,df_entropy
    
    @staticmethod
    def matrix_2_image(file,matrix):
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val != min_val:
            norm_matrix = (matrix - min_val) / (max_val - min_val) * 255
        else:
            norm_matrix = np.zeros_like(matrix)
        img_data = norm_matrix.astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
        img.save(file)
        
    @staticmethod
    def save_matrix(file,matrix):
        np.save(file, matrix)

    @staticmethod
    def load_matrix(file):
        return np.load(file+".npy",allow_pickle=True)

    def get_mfs(self,img_list = ['gray','r','g','b'],is_bin = False):
        if is_bin:
            df_mfs = self.m_mfs[self.m_mfs['bin'] == 'yes']
        else:
            df_mfs = self.m_mfs[self.m_mfs['bin'] == 'no']
        df_mfs = df_mfs[df_mfs['img'].isin(img_list)].reset_index(drop = True)
        value_list = ['t(q)', 'd(q)', 'a(q)', 'f(a)']
        pivot_df = df_mfs.pivot_table(index='r',columns='q',values=value_list,aggfunc='mean')
        result = []
        for value in value_list:
            matrix = pivot_df[value]
            result.append(matrix)
        return np.array(result)

    def get_lac(self,img_list = ['gray','r','g','b'],is_bin = False):
        if is_bin:
            df_lac = self.m_lac[self.m_lac['bin'] == 'yes']
        else:
            df_lac = self.m_lac[self.m_lac['bin'] == 'no']
        df_lac = df_lac[df_lac['img'].isin(img_list)].reset_index(drop = True)
        value_list = ['lacunarity']
        pivot_df = df_lac.pivot_table(index='r',columns='scales',values=value_list,aggfunc='mean')
        result = []
        for value in value_list:
            matrix = pivot_df[value]
            result.append(matrix)
        return np.array(result)

    def get_fd(self,img_list = ['gray','r','g','b'],is_bin = False):
        if is_bin:
            df_fd = self.m_fd[self.m_fd['bin'] == 'yes']
        else:
            df_fd = self.m_fd[self.m_fd['bin'] == 'no']
        df_fd = df_fd[df_fd['img'].isin(img_list)].reset_index(drop = True)
        result = []
        value_list = ['fd']
        pivot_df = df_fd.pivot_table(index='r',columns='img',values=value_list,aggfunc='mean')
        feature = []
        for value in value_list:
            matrix = pivot_df[value]
            result.append(matrix)
        return np.array(result)

file_list = [
        #"./data/face/dataset/deepfake/real/003_0265.png",
        "./data/face/dataset/train/fake/0.png"
        #"./data/face/dataset/train/real/0.png"
    ]

def show_images(images):
    sqrt_ceil = int(np.ceil(np.sqrt(len(images))))
    fig, axs = plt.subplots(sqrt_ceil, sqrt_ceil, figsize=(8, 8))
    axes = axs.flatten()
    i = 0
    for item in images:
        r = item['r']
        img = item['gray']
        axes[i].imshow(img, cmap='gray',vmin=0, vmax=255)
        axes[i].axis('off')
        axes[i].set_title('%0.2f'%r)
        i = i + 1
    plt.tight_layout()
    plt.show()

def save_images(file,df_fourier,folder = "./data/jim/fourier"):
    dataset = file.split("/")[-3]
    kind = file.split("/")[-2]
    file_name = file.split("/")[-1].split(".")[0]
    for item in df_fourier.to_dict(orient='records'):
        img = item['image']
        r = item['r']
        output_file = f"{folder}/{kind}/{file_name}-{r}.png"
        cv2.imwrite(output_file, img)

def get_file_name(file):
    dataset = file.split("/")[-3]
    kind = file.split("/")[-2]
    file_name = file.split("/")[-1].split(".")[0]
    return dataset+"-"+kind+"-"+file_name

def extract_feature():
    fd_list = []
    lac_list = []
    mfs_list = []
    for file in tqdm(file_list,desc= "total"): 
        file_name = get_file_name( file )
        df_fd,df_lac,df_mfs,df_fourier,df_entroy = CFeatureMatrix.extract(file,128,with_img=True,actions=['mfs'])
        df_fd['file'] = file_name
        df_lac['file']= file_name
        df_mfs['file']= file_name
        df_fourier['file'] = file_name

        fd_list.append(df_fd)
        lac_list.append(df_lac)
        mfs_list.append(df_mfs)
        if df_mfs.empty:
            print("failed",file_name)
        save_images(file,df_fourier)
    
    df_fd = pd.concat(fd_list,ignore_index = True).reset_index(drop = True)
    df_lac = pd.concat(lac_list,ignore_index = True).reset_index(drop = True)
    df_mfs = pd.concat(mfs_list,ignore_index = True).reset_index(drop = True)
    print(df_mfs)
    #df_fd.to_csv("./data/jim/fd.csv")
    #df_lac.to_csv("./data/jim/lac.csv")
    #df_mfs.to_csv("./data/jim/mfs.csv")   

def extract_matrix():
    df_fd = pd.read_csv("./data/jim/fd.csv",index_col=0)
    df_lac = pd.read_csv("./data/jim/lac.csv",index_col=0)
    df_mfs = pd.read_csv("./data/jim/mfs.csv",index_col=0)
    for file in tqdm(file_list,desc= "total"):     
        file_name =  get_file_name( file )
        test = CFeatureMatrix(df_fd[df_fd['file']==file_name],df_lac[df_lac['file']==file_name],df_mfs[df_mfs['file']==file_name])
        matrix = test.get_mfs(is_bin = False)
        print("FD:",test.get_fd(is_bin = True).shape)
        print("LAC:",test.get_lac(is_bin = True).shape)
        print("MFS:",test.get_mfs(is_bin = True).shape)
        plt.figure(figsize=(8, 6))
        heat_file_name = "./data/jim/matrix/%s-%s"%('mfs',file_name)
        CFeatureMatrix.save_matrix(heat_file_name,matrix)
        t = CFeatureMatrix.load_matrix(heat_file_name)
        print(t)

def show():
    df_data = pd.read_csv("./data/jim/fd.csv",index_col=0)
    df_tmp =  df_data[df_data['bin'] != 'yes']
    df_tmp = df_tmp.groupby(['r','file']).mean(numeric_only=True) 
    print(df_tmp)

    df_data = pd.read_csv("./data/jim/lac.csv", index_col=0)
    df_tmp = df_data[df_data['bin'] == 'no']
    df_tmp = df_tmp.groupby(['file','r']).mean(numeric_only=True) / df_tmp.groupby(['file','r']).std(numeric_only=True)
    print(df_tmp)

    df_data = pd.read_csv("./data/jim/mfs.csv", index_col=0)
    df_tmp = df_data[df_data['bin'] == 'no']
    df_tmp = df_tmp[df_tmp['img'] == 'gray']
    df_tmp = df_tmp.groupby(['file','r']).mean(numeric_only=True) #/ df_tmp.groupby(['file','r']).std(numeric_only=True)
    print(df_tmp)

if __name__ == "__main__":
    extract_feature()
    #extract_matrix()
    #show()
