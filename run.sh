#!/bin/bash
FEATURE_PATH='/home/signboard_retrieval/features' # layer_wise
QUERY_PATH='/home/signboard_retrieval/panorama_crop/q_crop_val' # orginal, keep_ratio
DB_PATH='/home/signboard_retrieval/panorama_crop/db_crop_val' 
IMAGE_SIZE=$1 # resolution_wise
IMAGE_MODE=$2
MODEL_MODE=$3
MODEL=$4 # model_wise
BATCH_SIZE=64
NUM_WORKERS=0
TOP_K_ALL=$5

python runtime_main.py \
    --feature_path $FEATURE_PATH \
    --q_img_path $QUERY_PATH \
    --db_img_path $DB_PATH \
    --image_size $IMAGE_SIZE \
    --image_mode $IMAGE_MODE \
    --model_mode $MODEL_MODE \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --top_k_all $TOP_K_ALL \