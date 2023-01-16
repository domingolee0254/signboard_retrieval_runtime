import argparse
import time
import faiss
import numpy as np
import torch
import matplotlib.pyplot as plt
import natsort
import os 
import itertools
import cv2
import csv
import pickle

from utils import metric

def search(args, query_feature, reference_feature):
    ### query feature extracting ### 
    start= time.time()
    query_feat = query_feature.numpy()
    faiss.normalize_L2(query_feat) # feature normalize_L2 for compare
    
    ### reference feature extracting ###
    reference_feat = reference_feature.numpy()
    faiss.normalize_L2(reference_feat) # feature normalize_L2 for compare
    print(f'[Load] {time.time() - start:.2f} sec, Query: {query_feat.shape}, Reference: {reference_feat.shape}') #print(query_feat.shape) #q.shape (100, 768) // db.shape (3377, 768)

    ### image feature similarity comparing ###  
    start = time.time()
    index = faiss.IndexFlatIP(reference_feat.shape[1])
    #index = faiss.IndexFlatL2(reference_feat.shape[1])
    index.add(reference_feat)
    D, I = index.search(query_feat, reference_feat.shape[0])
    print(f'[Search Time] {time.time() - start:.2f} sec')

    return D, I


def make_result(args, D, I, panorama_id):

    # q_imgs_path_list = []
    # db_imgs_path_list = []

    # ### 쿼리 이미지 인덱싱 ##
    # for root, dirs, files in os.walk(q_tmp_img_dir_idx):
    #     for file in files:
    #         q_imgs_path_list.append(os.path.join(q_tmp_img_dir_idx, file))
    # q_imgs_path_list = natsort.natsorted(set(q_imgs_path_list))
    
    # ### 디비 이미지 인덱싱 ##
    # for root, dirs, files in os.walk(db_tmp_img_dir_idx):
    #     for file in files:
    #         db_imgs_path_list.append(os.path.join(db_tmp_img_dir_idx, file))
    # db_imgs_path_list = natsort.natsorted(set(db_imgs_path_list))

    res_txt_path = f"result/vit_best_pair/pair_{panorama_id}_vit.txt"
    with open(res_txt_path, 'a', encoding='utf-8') as f:
        for n, q_idx in enumerate(range(I.shape[0])):
            #print(f"D is \n{D}")    # 두 대상의 유사도 높은 순으로 나열 
            #print(f"I is \n{I}\n")  # 유사도 높은 순서인 애들의 DB에서의 인덱스 
            #print([db_idx for i in range(I.shape[1])])
            for db_idx in range(I.shape[1]):
                print(f"=="*100)
                print(f"D[q_idx][db_idx is {D[q_idx][db_idx]}")
                #txt = q_imgs_path_list[q_idx].split('/')[-1]+','+db_imgs_path_list[I[q_idx][db_idx]].split('/')[-1]+','+str(D[q_idx][db_idx])
                txt = str(q_idx)+','+str(I[q_idx][db_idx])+','+str(D[q_idx][db_idx])
                print(f"D is {D}")
                print(f"=="*100)
                f.write(txt+'\n')

        
def main(args, query_feature, reference_feature, panorama_id):
    #img_dir_idx = query_path.split('/')[-2]
    #q_tmp_img_dir_idx = os.path.join(args.q_img_path, img_dir_idx)
    #q_imgs = os.listdir(q_tmp_img_dir_idx)

    D, I = search(args, query_feature, reference_feature)
    make_result(args, D, I, panorama_id)

if __name__ == '__main__':
    # feature extractor args 
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/home/signboard_retrieval/')
    parser.add_argument('--q_img_path', type=str, default='/home/signboard_retrieval/panorama_crop/q_crop_val/')
    parser.add_argument('--db_img_path', type=str, default='/home/signboard_retrieval/panorama_crop/db_crop_val/')
    parser.add_argument('--feature_path', type=str, default='/home/signboard_retrieval/features')
    parser.add_argument('--result_path', type=str, default='/home/signboard_retrieval/result')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    query_path = "/home/signboard_retrieval/features/q_crop_val/0/0_token_cls.pth"
    reference_path = "/home/signboard_retrieval/features/db_crop_val/0/0_token_cls.pth"
    
    main(args, query_path, reference_path)
