#!/bin/bash

# custom config

# Enter the path to your dataset
DATASET=$1

python saliency_maps/generate_saliency_maps.py \
--input-path ${DATASET}/images \
--output-path saliency_map_outputs/${DATASET}/masks \
--val-path ${DATASET}/val_images \
--model-name BiomedCLIP \
--finetuned \
--hyper-opt \
--val-path ${DATASET}/val_images

python postprocessing/postprocess_saliency_maps.py \
--input-path ${DATASET}/images \
--output-path coarse_outputs/${DATASET}/masks \
--sal-path saliency_map_outputs/${DATASET}/masks \
--postprocess kmeans \
--filter
# --num-contours 2 # number of contours to extract, for lungs, use 2 contours

python segment-anything/prompt_sam.py \
--input ${DATASET}/images \
--mask-input coarse_outputs/${DATASET}/masks \
--output sam_outputs/${DATASET}/masks \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes \
# --multicontour # for lungs, use this flag
