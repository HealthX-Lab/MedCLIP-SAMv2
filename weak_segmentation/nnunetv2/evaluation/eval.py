import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='data/breast/test_masks')
parser.add_argument('--seg_path', type=str, default='crf_outputs/breast/test_CRF')

args = parser.parse_args()
gt_path = args.gt_path
seg_path = args.seg_path

# Get list of filenames
filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

# Initialize metrics dictionary
seg_metrics = OrderedDict(
    Name = list(),
    DSC = list(),
    IoU = list(),
    NSD = list(),
)

# Compute metrics for each file
for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask = cv2.threshold(gt_mask, 200, 255, cv2.THRESH_BINARY)[1]
    seg_mask = cv2.threshold(seg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    gt_data = np.uint8(gt_mask)
    seg_data = np.uint8(seg_mask)

    gt_labels = np.unique(gt_data)[1:]
    seg_labels = np.unique(seg_data)[1:]
    labels = np.union1d(gt_labels, seg_labels)

    assert len(labels) > 0, 'Ground truth mask max: {}'.format(gt_data.max())

    DSC_arr = []
    IoU_arr = []
    NSD_arr = []
    for i in labels:
        if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
            DSC_i = 1
            IoU_i = 1
            NSD_i = 1
        elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
            DSC_i = 0
            IoU_i = 0
            NSD_i = 0
        else:
            i_gt, i_seg = gt_data == i, seg_data == i
            
            # Compute Dice coefficient
            DSC_i = compute_dice_coefficient(i_gt, i_seg)

            # Compute NSD
            case_spacing = [1, 1, 1]
            surface_distances = compute_surface_distances(i_gt[..., None], i_seg[..., None], case_spacing)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, 2)

        DSC_arr.append(DSC_i)
        NSD_arr.append(NSD_i)

    DSC = np.mean(DSC_arr)
    NSD = np.mean(NSD_arr)
    seg_metrics['DSC'].append(round(DSC, 4))
    seg_metrics['NSD'].append(round(NSD, 4))

# Save metrics to CSV
dataframe = pd.DataFrame(seg_metrics)

# Calculate and print average and std deviation for metrics
case_avg_DSC = dataframe['DSC'].mean()
case_avg_NSD = dataframe['NSD'].mean()

print(20 * '>')
print(f'Average DSC for {basename(seg_path)}: {case_avg_DSC}')
print(f'Average NSD for {basename(seg_path)}: {case_avg_NSD}')
print(20 * '<')
