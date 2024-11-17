#imports
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import cv2
import nnunetv2
from nnunetv2.nnunet.uncertainty_utils import *
import os
import shutil


def run_uncertainty_on_fold(proba_dir, raw_path,score_type , labels , output_pred_path = False):
    dice_list = []
    uncertainty_scors = []
    name_list = [name.split('.')[0].replace('_0000','') for name in os.listdir(raw_path)]

    for image_name in name_list:
        #compute p values map for the image
        #$ map all dir in the folder - those are the checkpoints.
        checkpoint_list = [checkpoint for checkpoint in os.listdir(proba_dir) if os.path.isdir(proba_dir + '/' + checkpoint)]

        # checkpoint_list = ['checkpoint_289', 'checkpoint_290', 'checkpoint_291', 'checkpoint_292', 'checkpoint_293', 'checkpoint_294', 'checkpoint_295', 'checkpoint_296',
        #                     'checkpoint_297', 'checkpoint_298','checkpoint_best', 'checkpoint_final']

        class0_array = [] #$ background
        class1_array = [] #$ forground
        #$ for each checkpoint we will have a list of probability maps for each class.

        for checkpoint in checkpoint_list:
            #$ Load probs from .npz file
            prediction_file = np.load(proba_dir + '/' + checkpoint + '/' + image_name + '.npz', allow_pickle=True)
            #$ append the probability map of each class to the class array.
            class0_array.append(prediction_file['probabilities'][0, 0, :, :])
            class1_array.append(prediction_file['probabilities'][1, 0, :, :])

        #$ convert the class arrays to numpy arrays.
        class0_array = np.array(class0_array)
        class1_array = np.array(class1_array)
        # Threshold for Breast Pseudo is 0.2
        mask = (np.mean(class1_array, axis=0) > 0.2).astype(np.uint8)
        map = np.zeros_like(mask)
        if score_type == 't_test':
            p_values_map = T_test_on_single_image(class0_array, class1_array, plot_results = False)
            uncertainty_score = uncertainty_from_mask_and_valmap(p_values_map ,  mask)
            map = p_values_map

        elif score_type == 'class_entropy':
            class_entropy_map = entropy_map_fun(np.mean(class1_array,axis = 0), np.mean(class0_array,axis = 0))
            uncertainty_score =  uncertainty_from_mask_and_valmap(class_entropy_map ,  mask)
            map = class_entropy_map
        elif score_type == 'total_entropy':
            #append class one and class two on axis 0
            np.concatenate((class0_array, class1_array), axis = 0)
            total_entropy_map = -np.sum(class0_array * np.log(class0_array), axis=0)
            uncertainty_score =  uncertainty_from_mask_and_valmap(total_entropy_map ,  mask)
            map = total_entropy_map

        uncertainty_scors.append(uncertainty_score)
        if labels:
            # label =  load_niigii_file(labels + '/' + image_name + '.png')
            label = cv2.imread(labels + '/' + image_name + '.png', cv2.IMREAD_GRAYSCALE)
            temp_dice =   dice(mask , label)
            dice_list.append(temp_dice)
        
        if not output_pred_path:
            output_pred_path = proba_dir + '/unnunet_pred'
        if not os.path.exists(output_pred_path):
            os.makedirs(output_pred_path)

        # #copy prediction mask
        # predicted_mask = proba_dir + '/checkpoint_best/' + image_name + '.png'
        # #copy prediction mask to output folder
        # shutil.copy(predicted_mask, output_pred_path + '/' + image_name + '_predicted_mask.png')

        cv2.imwrite(output_pred_path + '/' + image_name + '.png', mask)
        
        #save the uncertainty map
        # map_nii = nib.Nifti1Image(map, np.eye(4))
        # normalized_map = (map - np.min(map)) / (np.max(map) - np.min(map)) * 255
        # normalized_map = normalized_map.astype(np.uint8)
        # cv2.imwrite(output_pred_path + '/' + image_name + '_uncertainty_map.png', normalized_map)
        # nib.save(map_nii, output_pred_path + '/' + image_name + '_uncertainty_map.png')
    #save the uncertainty scores, with the image names , if dice availible also dice scores.
    uncertainty_df = pd.DataFrame({'image_name': name_list, 'uncertainty_score': uncertainty_scors})
    if labels:
        uncertainty_df['dice_score'] = dice_list
    uncertainty_df.to_csv(output_pred_path + '/uncertainty_scores.csv', index=False)
    return uncertainty_df

        

def run_uncertainty_on_fold_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proba_dir', type=str, default='', help='path to the folder with the checkpoints folders (output of previous script)')
    parser.add_argument('--raw_path', type=str, default='', help='path to the folder with the dataset the user wants to predict ( input of previous script)')
    parser.add_argument('--labels', type=str, default='', help='optional - path to the labels of the dataset')
    parser.add_argument('--score_type', type=str, default='class_entropy', help='optional - the score type to use for the uncertainty score. default is class_entropy - other options are total_entropy and t_test')
    parser.add_argument('--output_pred_path', type=str, default='', help='optional - path to the folder where the predictions will be saved. default is proba_dir + /unnunet_pred')
    args = parser.parse_args()

    
    run_uncertainty_on_fold(args.proba_dir, args.raw_path, args.score_type , args.labels , args.output_pred_path)

if __name__ == '__main__':
    run_uncertainty_on_fold_entry()


