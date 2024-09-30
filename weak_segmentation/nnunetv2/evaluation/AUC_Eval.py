import numpy as np
#import nibabel as nb
import cv2
import os
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import roc_auc_score
join = os.path.join
basename = os.path.basename
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_path',
    type=str,
    default=''
)
parser.add_argument(
    '--seg_path',
    type=str,
    default=''
)
parser.add_argument(
    '--save_path',
    type=str,
    default=''
)

args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.png')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

seg_metrics = OrderedDict(
    Name = list(),
    AUC = list(),
)

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)


    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (seg_mask.shape[1], seg_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    case_spacing = [1,1,1]
    gt_data = np.uint8(gt_mask)
    seg_data = np.uint8(seg_mask)

    gt_labels = np.unique(gt_data)[1:]
    seg_labels = np.unique(seg_data)[1:]
    labels = np.union1d(gt_labels, seg_labels)

    assert len(labels) > 0, 'Ground truth mask max: {}'.format(gt_data.max())

    AUC_arr = []
    for i in labels:

        tool_i_gt, tool_i_seg = gt_data==i, seg_data==i
        AUC_i = roc_auc_score(tool_i_gt.flatten().astype(float), tool_i_seg.flatten().astype(float))

        AUC_arr.append(AUC_i)

    AUC = np.mean(AUC_arr)
    seg_metrics['AUC'].append(round(AUC, 4))

dataframe = pd.DataFrame(seg_metrics)
#dataframe.to_csv(seg_path + '_DSC.csv', index=False)
dataframe.to_csv(save_path, index=False)

case_avg_AUC = dataframe.mean(axis=0, numeric_only=True)
case_std_AUC = dataframe.std(axis=0, numeric_only=True)
print(20 * '>')
print(f'Average AUC for {basename(seg_path)}: {case_avg_AUC.mean()}')
print(f'Standard deviation AUC for {basename(seg_path)}: {case_std_AUC[0]}')
print(20 * '<')