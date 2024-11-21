#!/bin/bash

# custom config

# Enter the path to your dataset
DATASET=$1

python saliency_maps/generate_saliency_maps.py \
--input-path ${DATASET}/test_images \
--output-path saliency_map_outputs/${DATASET}/test_masks \
--model-name BiomedCLIP \
--finetuned \
--json-path saliency_maps/text_prompts/breast_tumors_testing.json \
--reproduce \
--vvar 1.0 \
--vbeta 1.0 \
--vlayer 9 \
--seed 12

python postprocessing/postprocess_saliency_maps.py \
--input-path ${DATASET}/test_images \
--output-path coarse_outputs/${DATASET}/test_masks \
--sal-path saliency_map_outputs/${DATASET}/test_masks \
--postprocess kmeans \
--filter

python segment-anything/prompt_sam.py \
--input ${DATASET}/test_images \
--mask-input coarse_outputs/${DATASET}/test_masks \
--output sam_outputs/${DATASET}/test_masks \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes

python evaluation/eval.py \
--gt_path ${DATASET}/test_masks \
--seg_path sam_outputs/${DATASET}/test_masks
