import argparse
import os
import natsort
import pickle5 as pickle

import feature_extractor
import rank

from utils.make_file import find_files #__init__.py 로 다시 수정해보기 

def main(args, q_panorama_list, db_panorama_list, q_crop_list, db_crop_list, panorama_id):
    
    query_feature, reference_feature = feature_extractor.main(args, q_crop_list, db_crop_list) #파노라마별 크롭된 이미지가 리스트
    rank.main(args, query_feature, reference_feature, panorama_id)
    # q_feat_path_list = []
    # db_feat_path_list = []

    # ### 쿼리 이미지 파일 경로 반환 ###
    # for root, dirs, files in os.walk(os.path.join(args.feature_path, os.path.basename(args.q_img_path))):
    #     for file in files:
    #         if file.endswith('.pth'):
    #             q_feat_path_list.append(os.path.join(root, file))
    # q_feat_path_list = natsort.natsorted(set(q_feat_path_list))

    # # # ### 레퍼런스 이미지 파일 경로 반환 ###
    # for root, dirs, files in os.walk(os.path.join(args.feature_path, os.path.basename(args.db_img_path))):
    #     for file in files:
    #         if file.endswith('.pth'):
    #             db_feat_path_list.append(os.path.join(root, file))
    # db_feat_path_list = natsort.natsorted(set(db_feat_path_list))

if __name__ == '__main__':
    # feature extractor args 
    parser = argparse.ArgumentParser(description='Extract frame feature') 
    parser.add_argument('--result_path', type=str, default='/home/signboard_retrieval/result')
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')   
    args = parser.parse_args()

    ###나중에 삭제##
    with open('/home/signboard_retrieval/pkl/db_panorama_list.pkl', 'rb') as fp: #len은 동일
        db_panorama_list = pickle.load(fp)
    with open('/home/signboard_retrieval/pkl/q_panorama_list.pkl', 'rb') as fp: #len은 동일
        q_panorama_list = pickle.load(fp)
    with open('/home/signboard_retrieval/pkl/db_crop_list.pkl', 'rb') as fp: #len은 동일
        db_crop_list = pickle.load(fp)
    with open('/home/signboard_retrieval/pkl/q_crop_list.pkl', 'rb') as fp: #len은 동일
        q_crop_list = pickle.load(fp)
    
    panorama_id = 0
    main(args, q_panorama_list, db_panorama_list, q_crop_list, db_crop_list, panorama_id)
    
