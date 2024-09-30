import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Union, Tuple
import torch
from os.path import join
import nibabel as nib



def get_files_from_folder(folder_path):
    """
    Retrieves a list of file paths for files with a '.nii.gz' extension within a folder and its subfolders.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of file paths.

    """
    files = []  # List to store file paths

    # Recursively walk through the folder and its subfolders
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.nii.gz'):  # Check if the file has a '.nii.gz' extension
                file_path = os.path.join(root, filename)  # Create the full file path by joining the root and filename

                files.append(file_path)  # Add the file path to the list

    return files


def dice_coefficient(pred_labels, true_labels):
    """
    Calculates the Dice coefficient, a measure of similarity, between two binary label volumes.
    Note: the dice output is per image and per checkpoint!!!
    Args:
        pred_labels (nibabel.nifti1.Nifti1Image): Predicted label volume as a NIfTI image object.
        true_labels (nibabel.nifti1.Nifti1Image): True label volume as a NIfTI image object.

    Returns:
        float: Dice coefficient value between 0.0 and 1.0, indicating the similarity between the two volumes.

    """
    pred_data = pred_labels.astype(bool)  # Extract the data array from the predicted label volume and convert it to boolean
    true_data = true_labels.get_fdata().astype(bool)  # Extract the data array from the true label volume and convert it to boolean

    intersection = np.sum(pred_data & true_data)  # Compute the sum of the element-wise logical AND operation between the predicted and true label data arrays
    sum_pred = np.sum(pred_data)  # Compute the sum of the predicted label data array
    sum_true = np.sum(true_data)  # Compute the sum of the true label data array

    dice = (2.0 * intersection) / (sum_pred + sum_true)  # Calculate the Dice coefficient using the intersection and sum values

    return dice

def dice_coeff_all_im(all_predictions_mean, True_lables_paths):
    """
    Calculate the Dice coefficients for all predicted labels compared to their corresponding true labels.

    Args:
        all_predictions_mean (list): List of predicted images (was taken the mean of all checkpoints).
        True_lables_paths (list): List of paths to the true labels.

    Returns:
        dice_coefficients (list): List of Dice coefficients for each predicted label.

    """
    dice_coefficients = []
    for pred_labels, true_labels in zip(all_predictions_mean, True_lables_paths):
        true_seg = nib.load(true_labels)
        dice_coeff = dice_coefficient(pred_labels, true_seg)
        dice_coefficients.append(dice_coeff)

    return dice_coefficients


def get_Antropy_all_weights(checkpoint_list, folder_path, pred):
     """
    Retrieves the probability map for a single image from a folder containing multiple checkpoints.
    Note: this is publication method!!
    Args:
        checkpoint_list

    Returns:
        numpy.ndarray: The entropy map as a 2D NumPy array.

    """
    #map all dir in the folder
 
     probabilities = []
     for checkpoint in checkpoint_list:
        # Load probs from .npz file
        prediction_file = np.load(folder_path + '/' + checkpoint + '/' + pred + '.npz', allow_pickle=True)
        probabilities.append(prediction_file['probabilities'][1, 0, :, :])
        
     probability_maps = np.array(probabilities)
     # Calculate entropy function across dim=0
     entropy_map = -np.sum(probability_maps * np.log2(probability_maps), axis=0)
    
     return entropy_map 

def get_mean_probability_map_for_single_image(checkpoint_list,folder_path,pred):
    """
    Retrieves the mean probability maps for foreground and background for a single image from a folder containing multiple checkpoints.

    Args:
        folder_path (str): The path to the folder.
        pred (str): The prediction name.

    Returns:
        numpy.ndarray: The mean probability map for the foreground as a 2D NumPy array.
        numpy.ndarray: The mean probability map for the background as a 2D NumPy array.

    """
        
    probabilities_fg = [] #forground
    probabilities_bg = [] #background

    for checkpoint in checkpoint_list:
        # Load probs from .npz file
        prediction_file = np.load(folder_path + '/' + checkpoint + '/' + pred + '.npz', allow_pickle=True)
        probabilities_fg.append(prediction_file['probabilities'][1, 0, :, :])
        probabilities_bg.append(prediction_file['probabilities'][0, 0, :, :])

    #calculate mean of probabilities
    probability_maps_fg = np.mean(probabilities_fg,axis = 0)
    probability_maps_bg = np.mean(probabilities_bg,axis = 0)
    
    return probability_maps_fg , probability_maps_bg

def entropy_map_fun(fg_map, bg_map):

    # Calculate entropy function across dim=0
    entropy_map = -(fg_map * np.log2(fg_map) + bg_map* np.log2(bg_map))
    
    #count uncertainty
    
    return entropy_map 

def uncertainty_score_dilaton_normalized(map: np.ndarray, mask: np.ndarray , plot: bool = False ):
    """
    Calculates the uncertainty score based on a probability map and a corresponding mask.

    Args:
        map (numpy.ndarray): Probability map as a 2D NumPy array.
        mask (numpy.ndarray): Mask as a 2D NumPy array.
        plot (bool, optional): Whether to plot the results. Defaults to False.
        image (numpy.ndarray, optional): Image as a 2D NumPy array. Required if plot=True. Defaults to None.

    Returns:
        float: Uncertainty score.

    """
    #$ Perform dilation operation
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    #$ Get the edges by subtracting the dilated mask from the original mask
    edge_mask = abs(mask - dilated_mask)

    #$ sum all values in p_values_map and divide by the number of pixels in the edge mask
    try:
        certainty_score = 100 * np.sum(map) / ((np.sum(edge_mask)))
    except:
        certainty_score = 0
    
    return certainty_score 

#$ for wach class we have [num of checkpiints] probability maps, thas we can run pixelwise T test.
#$ conduct T test on the probability maps of the same image for different labels.
#$ the T test conducted does not assume equal variance between the two groups.
#$ we first adress the option of only two classes, maybe later we will adress more classes. - but probably we will need to use ANOVA.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

#utilities for loading files
def load_niigii_file(file_path , plot:bool = False):
    data = nib.load(file_path)
    image = data.get_fdata()
    if plot:
        plt.imshow(image, cmap='gray')
        plt.show()
    return image

def load_npz_file(file_path , plot:bool = False):
    data = np.load(file_path, allow_pickle=True)
    probabilities = data['probabilities']
    if plot:
        plt.imshow(probabilities[0, 0, :, :] , cmap='gray')
        plt.show()
    return probabilities

#dice
def dice(pred_array, label_array):
    intersection = np.sum(pred_array * label_array)
    sum_pred = np.sum(pred_array)
    sum_true = np.sum(label_array)

    dice = (2.0 * intersection) / (sum_pred + sum_true)

    return dice

#$ this functuin gets two arrays of the same size and conduct T test on them on axis 0, pixelwise.
#$ you can also plot results if you set the plot_results flag to True, don't forget to set a title.
def T_test_on_single_image(class1_array: np.ndarray, class2_array: np.ndarray ,title:str = 'p_values_map' , plot_results: bool = False):
    
    # Perform t-test with unequal variances along the first dimension
    t_values, p_values_map = stats.ttest_ind(class1_array, class2_array, axis=0, equal_var=False)
    
    if plot_results:
        # Plot the p-value map
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(p_values_map, cmap='viridis')
        ax.set_title( title )
        #plt.colorbar(fig)
        plt.show()
    return p_values_map.T

#$ this function gets a p_values_map and mask and returns uncertainty level.
#$ plot needs the original image as input!
def uncertainty_from_mask_and_valmap(p_values_map: np.ndarray, mask: np.ndarray , plot: bool = False , image: np.ndarray = None):
    #$ Perform dilation operation
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    #$ Get the edges by subtracting the dilated mask from the original mask
    edge_mask = abs(mask - dilated_mask)

    #$ sum all values in p_values_map and divide by the number of pixels in the edge mask
    try:
        certainty_score = 100 * np.sum(p_values_map) / (np.sum(edge_mask))
    except:
        certainty_score = 0
    
    if plot:
        #need the image!!
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(p_values_map, cmap='gray')
        ax[0].set_title('Pvals map + edges , score: ' + str(certainty_score))
        ax[0].imshow(edge_mask, cmap='gray',alpha=0.2)
        ax[1].imshow(image, cmap='gray')
        ax[1].imshow(mask, cmap='gray', alpha=0.5)
        ax[1].set_title('image with mask')
        plt.show()
    
    return certainty_score 

#$ this function gets a folder path of the fold (dir of the checkpoint folders), computes the p_values_map, returns the mask and uncertainty level. 
#$ images are saved in the father directory of the fold path in a directory named 't_test_masks_fold[number]'. 
#$ make sure to enter path without / at the end.
def build_prob_arrays_for_a_given_image(fold_path: str, save_p_val_maps: bool = False, threshold: float = 0.05 ,save_all_arrays: bool = False):
    #$ map all dir in the folder - those are the checkpoints.
    checkpoint_list = [checkpoint for checkpoint in os.listdir(fold_path) if os.path.isdir(fold_path + '/' + checkpoint)]
    #$ map all image names inside the first checkpoint dir
    images_list = [x.replace('.nii.gz',"") for x in os.listdir(fold_path + '/' + checkpoint_list[0]) if x.endswith('.nii.gz')]

    #$ use a nested loop to access each checkpoint and each image in the checkpoint.
    for image_name in images_list:
        #$ for each image we will have a list of probability maps for each class.
        class0_array = [] #$ background
        class1_array = [] #$ forground
        #$ for each checkpoint we will have a list of probability maps for each class.
        for checkpoint in checkpoint_list:
            #$ Load probs from .npz file
            prediction_file = np.load(fold_path + '/' + checkpoint + '/' + image_name + '.npz', allow_pickle=True)
            #$ append the probability map of each class to the class array.
            class0_array.append(prediction_file['probabilities'][0, 0, :, :])
            class1_array.append(prediction_file['probabilities'][1, 0, :, :])

        #$ convert the class arrays to numpy arrays.
        class0_array = np.array(class0_array)
        class1_array = np.array(class1_array)

        #$ conduct T test on the probability maps of the same image for different labels.
        p_values_map = T_test_on_single_image(class0_array, class1_array, title = image_name + ' p_values_map', plot_results = False).T
        
        #$ get uncertainty level from the p_values_map.
        #mask used forucertainty calculation is the mean of the forground probability maps.
        mask = (np.mean(class1_array, axis=0).T > 0.5).astype(np.uint8) 
        uncertainty_score = uncertainty_from_mask_and_p_values(p_values_map, mask)

        #$ save the mask and uncertainty level in the father directory of the fold path in a directory named 't_test_masks_fold[number]'.
        if not os.path.exists(os.path.dirname(fold_path) + '/' + 't_test_masks_' + os.path.basename(fold_path)):
            os.mkdir(os.path.dirname(fold_path) + '/' + 't_test_masks_' + os.path.basename(fold_path))
        
        if save_all_arrays:
            np.save(os.path.dirname(fold_path) + '/' + 't_test_masks_' + os.path.basename(fold_path) + '/' + image_name + '_mask.npy', mask)
            np.save(os.path.dirname(fold_path) + '/' + 't_test_masks_' + os.path.basename(fold_path) + '/' + image_name + '_uncertainty_level.npy', uncertainty_score)
        if  save_p_val_maps:
            np.save(os.path.dirname(fold_path) + '/' + 't_test_masks_' + os.path.basename(fold_path) + '/' + image_name + '_p_values_map.npy', p_values_map)
