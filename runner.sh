#!/bin/bash

# #### runtime ####
bash run.sh 384 'original' 'ATTN' 'swin_large_patch4_window12_384_in22k' all

# #### dev ####
# ### 224 ###
# ## CNN ##
# # original # 
# bash run.sh 224 'original' 'CNN' 'efficientnet_b4' 1
# bash run.sh 224 'original' 'CNN' 'efficientnet_b4' 10
# bash run.sh 224 'original' 'CNN' 'cspresnet50' 1
# bash run.sh 224 'original' 'CNN' 'cspresnet50' 10
# bash run.sh 224 'original' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 1
# bash run.sh 224 'original' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 10
# bash run.sh 224 'original' 'CNN' 'regnetz_040h' 1
# bash run.sh 224 'original' 'CNN' 'regnetz_040h' 10
# bash run.sh 224 'original' 'CNN' 'ssl_resnet50' 1
# bash run.sh 224 'original' 'CNN' 'ssl_resnet50' 10
# bash run.sh 224 'original' 'CNN' 'resnetv2_152x4_bitm_in21k' 1
# bash run.sh 224 'original' 'CNN' 'resnetv2_152x4_bitm_in21k' 10
# bash run.sh 224 'original' 'CNN' 'convnext_xlarge_in22k' 1
# bash run.sh 224 'original' 'CNN' 'convnext_xlarge_in22k' 10

# # blackboarder #                 
# bash run.sh 224 'keep_latio' 'CNN' 'efficientnet_b4' 1
# bash run.sh 224 'keep_latio' 'CNN' 'efficientnet_b4' 10
# bash run.sh 224 'keep_latio' 'CNN' 'cspresnet50' 1
# bash run.sh 224 'keep_latio' 'CNN' 'cspresnet50' 10
# bash run.sh 224 'keep_latio' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 1
# bash run.sh 224 'keep_latio' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 10
# bash run.sh 224 'keep_latio' 'CNN' 'regnetz_040h' 1
# bash run.sh 224 'keep_latio' 'CNN' 'regnetz_040h' 10
# bash run.sh 224 'keep_latio' 'CNN' 'ssl_resnet50' 1
# bash run.sh 224 'keep_latio' 'CNN' 'ssl_resnet50' 10
# bash run.sh 224 'keep_latio' 'CNN' 'resnetv2_152x4_bitm_in21k' 1
# bash run.sh 224 'keep_latio' 'CNN' 'resnetv2_152x4_bitm_in21k' 10
# bash run.sh 224 'keep_latio' 'CNN' 'convnext_xlarge_in22k' 1
# bash run.sh 224 'keep_latio' 'CNN' 'convnext_xlarge_in22k' 10


## ATTN ##
# original # 
# bash run.sh 224 'original' 'ATTN' 'vit_base_patch8_224_dino' 1
# bash run.sh 224 'original' 'ATTN' 'vit_base_patch8_224_dino' 10
# bash run.sh 224 'original' 'ATTN' 'swin_large_patch4_window7_224_in22k' 1
# bash run.sh 224 'original' 'ATTN' 'swin_large_patch4_window7_224_in22k' 10
# bash run.sh 224 'original' 'ATTN' 'beitv2_large_patch16_224_in22k' 1
# bash run.sh 224 'original' 'ATTN' 'beitv2_large_patch16_224_in22k' 10

# # # blackboarder #                 
# bash run.sh 224 'keep_latio' 'ATTN' 'vit_base_patch8_224_dino' 1
# bash run.sh 224 'keep_latio' 'ATTN' 'vit_base_patch8_224_dino' 10
# bash run.sh 224 'keep_latio' 'ATTN' 'swin_large_patch4_window7_224_in22k' 1
# bash run.sh 224 'keep_latio' 'ATTN' 'swin_large_patch4_window7_224_in22k' 10
# bash run.sh 224 'keep_latio' 'ATTN' 'beitv2_large_patch16_224_in22k' 1
# bash run.sh 224 'keep_latio' 'ATTN' 'beitv2_large_patch16_224_in22k' 10


# ### 384 ###
# ## CNN ##
# # original # 
# bash run.sh 384 'original' 'CNN' 'efficientnet_b4' 1
# bash run.sh 384 'original' 'CNN' 'efficientnet_b4' 10
# bash run.sh 384 'original' 'CNN' 'cspresnet50' 1
# bash run.sh 384 'original' 'CNN' 'cspresnet50' 10
# bash run.sh 384 'original' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 1
# bash run.sh 384 'original' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 10
# bash run.sh 384 'original' 'CNN' 'regnetz_040h' 1
# bash run.sh 384 'original' 'CNN' 'regnetz_040h' 10
# bash run.sh 384 'original' 'CNN' 'ssl_resnet50' 1
# bash run.sh 384 'original' 'CNN' 'ssl_resnet50' 10
# bash run.sh 384 'original' 'CNN' 'resnetv2_152x4_bitm_in21k' 1
# bash run.sh 384 'original' 'CNN' 'resnetv2_152x4_bitm_in21k' 10
# bash run.sh 384 'original' 'CNN' 'convnext_xlarge_in22k' 1
# bash run.sh 384 'original' 'CNN' 'convnext_xlarge_in22k' 10

# # blackboarder #                 
# bash run.sh 384 'keep_latio' 'CNN' 'efficientnet_b4' 1
# bash run.sh 384 'keep_latio' 'CNN' 'efficientnet_b4' 10
# bash run.sh 384 'keep_latio' 'CNN' 'cspresnet50' 1
# bash run.sh 384 'keep_latio' 'CNN' 'cspresnet50' 10
# bash run.sh 384 'keep_latio' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 1
# bash run.sh 384 'keep_latio' 'CNN' 'efficientnet_b5.in12k_ft_in1k' 10
# bash run.sh 384 'keep_latio' 'CNN' 'regnetz_040h' 1
# bash run.sh 384 'keep_latio' 'CNN' 'regnetz_040h' 10
# bash run.sh 384 'keep_latio' 'CNN' 'ssl_resnet50' 1
# bash run.sh 384 'keep_latio' 'CNN' 'ssl_resnet50' 10
# bash run.sh 384 'keep_latio' 'CNN' 'resnetv2_152x4_bitm_in21k' 1
# bash run.sh 384 'keep_latio' 'CNN' 'resnetv2_152x4_bitm_in21k' 10
# bash run.sh 384 'keep_latio' 'CNN' 'convnext_xlarge_in22k' 1
# bash run.sh 384 'keep_latio' 'CNN' 'convnext_xlarge_in22k' 10

# # ATTN ##
# original # 
# bash run.sh 384 'original' 'ATTN' 'vit_base_patch32_384' 1
# bash run.sh 384 'original' 'ATTN' 'vit_base_patch32_384' 10
# bash run.sh 384 'original' 'ATTN' 'vit_base_r50_s16_384' 1
# bash run.sh 384 'original' 'ATTN' 'vit_base_r50_s16_384' 10
# bash run.sh 384 'original' 'ATTN' 'swin_large_patch4_window12_384' 1
# bash run.sh 384 'original' 'ATTN' 'swin_large_patch4_window12_384' 10
# bash run.sh 384 'original' 'ATTN' 'swin_large_patch4_window12_384_in22k' 1
# bash run.sh 384 'original' 'ATTN' 'swin_large_patch4_window12_384_in22k' 10
# bash run.sh 384 'original' 'ATTN' 'beit_large_patch16_384' 1
# bash run.sh 384 'original' 'ATTN' 'beit_large_patch16_384' 10

# # blackboarder #                 
# bash run.sh 384 'keep_latio' 'ATTN' 'vit_base_patch32_384' 1
# bash run.sh 384 'keep_latio' 'ATTN' 'vit_base_patch32_384' 10
# bash run.sh 384 'keep_latio' 'ATTN' 'vit_base_r50_s16_384' 1
# bash run.sh 384 'keep_latio' 'ATTN' 'vit_base_r50_s16_384' 10
# bash run.sh 384 'keep_latio' 'ATTN' 'swin_large_patch4_window12_384' 1
# bash run.sh 384 'keep_latio' 'ATTN' 'swin_large_patch4_window12_384' 10
# bash run.sh 384 'keep_latio' 'ATTN' 'swin_large_patch4_window12_384_in22k' 1
# bash run.sh 384 'keep_latio' 'ATTN' 'swin_large_patch4_window12_384_in22k' 10
# bash run.sh 384 'keep_latio' 'ATTN' 'beit_large_patch16_384' 1
# bash run.sh 384 'keep_latio' 'ATTN' 'beit_large_patch16_384' 10