import argparse
import os
import time
import faiss
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import natsort
import pickle5 as pickle
import numpy as np

from torch.utils.data import  DataLoader
from torchvision import models 
from torchinfo import summary
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm

from dataset import AugmentedDataset
from model import vit_base_patch8_224_dino, beitv2_large_patch16_224_in22k, swin_large_patch4_window12_384_in22k

def find_files(args):
    q_idx_path_list = []
    db_idx_path_list = []

    ### 쿼리 이미지 파일 경로 반환 ###
    for root, dirs, files in os.walk(args.q_img_path):
        for file in files:
            if file.endswith('.jpg'):
                q_idx_path_list.append(os.path.join(root))
    q_idx_path_list = natsort.natsorted(set(q_idx_path_list))
    
    ### 레퍼런스 이미지 파일 경로 반환 ###
    for root, dirs, files in os.walk(args.db_img_path):
        for file in files:
            if file.endswith('.jpg'):
                db_idx_path_list.append(os.path.join(root))
    db_idx_path_list = natsort.natsorted(set(db_idx_path_list))

    return q_idx_path_list, db_idx_path_list

@torch.no_grad()
def features_extract(args, model, data_idx_path):
    features = []

    dataset = AugmentedDataset(data_idx_path, img_size=384)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    bar = tqdm(loader, ncols=120, desc='extracting', unit='batch')
    
    start = time.time()
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(args.device)
        feat = model(imgs).cpu()
        print(f'feat shape is {feat.shape}')
        features.append(feat)
    print(feat.shape[0])
    print(f'feature extraction: {time.time() - start:.2f} sec')

    start = time.time()
    feature = np.vstack(features)
    print(f"feature shape is {feature.shape}")

    feature = feature.reshape(feature.shape[0],-1)
    print(f"new_feature shape is {feature.shape}")
    
    print(f'convert to numpy: {time.time() - start:.2f} sec')

    start = time.time()
    feature = torch.from_numpy(feature)
    print(f'convert to tensor: {time.time() - start:.2f} sec')

    start = time.time()
    
    return feature

class FeatLayer:
    def __init__(self):
        self.model = model

    def build_model(args, model, q_crop_list, db_crop_list):
        print(f"{summary(model, input_size=(1, 3, 384, 384))}")
        ############# device check #############
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(args.device)

        ############ 쿼리/ 레퍼런스 이미지 인덱스 폴더별 피처뽑기 #######
        data_types = [q_crop_list, db_crop_list]
        for i, pan_img_dir_list in enumerate(data_types):
            #print(f"i is {i}") # 1,2
            #print(f"pan_img_dir_list is {pan_img_dir_list}")
            if data_types[0]==pan_img_dir_list:
                query_feature = features_extract(args, model, pan_img_dir_list)
                print(f"===== Done: query_feature\n\n")
            else:
                reference_feature = features_extract(args, model, pan_img_dir_list)
                print(f"===== Done: db_feature\n\n")
        
        return query_feature, reference_feature

############# create ATTN model #############
def attn(args, q_crop_list, db_crop_list):    
   
    model = swin_large_patch4_window12_384_in22k()
    query_feature, reference_feature = FeatLayer.build_model(args, model, q_crop_list, db_crop_list)
    
    del model
    
    return  query_feature, reference_feature

def main(args, q_crop_list, db_crop_list):

    query_feature, reference_feature = attn(args, q_crop_list, db_crop_list)

    return query_feature, reference_feature
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/home/signboard_retrieval/')
    parser.add_argument('--q_img_path', type=str, default='/home/signboard_retrieval/panorama_crop/q_crop_val')
    parser.add_argument('--db_img_path', type=str, default='/home/signboard_retrieval/panorama_crop/db_crop_val')
    parser.add_argument('--feature_path', type=str, default='/home/signboard_retrieval/features')
    parser.add_argument('--result_path', type=str, default='/home/signboard_retrieval/result')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')    
    args = parser.parse_args()
