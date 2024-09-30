#imports
import os
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from time import sleep
import nnunetv2
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


# this function predicts multiple checkpoints from a trained folder
def predict_from_folder(dataset, fold:int , indir , outdir , rule = 'both'):
    #dataset: name or id of the dataset
    #folds: the fold to use for prediction
    #indir: path to the folder with the images to predict
    #outdir: path to the folder where the predictions will be saved
    #rule: 'late' or 'sparse' or 'both' - which checkpoints to predict

    #load variables
    nnUNet_raw = os.environ.get('nnUNet_raw')
    nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
    nnUNet_results = os.environ.get('nnUNet_results')

    if nnUNet_raw is None:
        print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
            "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
            "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
            "this up properly.")

    if nnUNet_preprocessed is None:
        print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
            "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
            "to set this up.")

    if nnUNet_results is None:
        print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
            "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
            "on how to set this up.")
        
    # provide relevant folder names for prediction and weights exctraction:
    Data_set_name = maybe_convert_to_dataset_name(dataset)
    Data_set_name = Data_set_name + "/nnUNetTrainer__nnUNetPlans__2d/"

    # initiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # creat an array of all loaded checkpoints

    checkpoint_list = [checkpoint for checkpoint in os.listdir(join(nnUNet_results, Data_set_name + 'fold_' + str(fold))) if checkpoint.endswith('.pth')]
    print('checkpoints in folder:')
    print(checkpoint_list)
    print('--------------------------------------')

    #filter acording to rule
    if rule == 'late':
        #take only checkpoints that sart with 'checkpoint_'
        checkpoint_list = [checkpoint for checkpoint in checkpoint_list if checkpoint.startswith('checkpoint_')]
        
    elif rule == 'sparse':
        # take only checkpoints that sart with 'e_checkpoint_'
        checkpoint_list = [checkpoint for checkpoint in checkpoint_list if checkpoint.startswith('e_checkpoint_')]
    elif rule == 'both':
        #do nothing
        checkpoint_list = checkpoint_list

    #add the best checkpoint to the list
    checkpoint_list.append('checkpoint_best.pth')
    print('rule = ' +str(rule) + ' //  checkpoints used: ')
    print(str(checkpoint_list))
    print('--------------------------------------')    
    

    if not indir.endswith('/'):
        indir = indir + '/'
    

    #load checkpoints and save probabilities:
    for checkpoint in checkpoint_list:  
        print("-----------------------------------")
        print("predicting from checkpoint: " + checkpoint)
        print("-----------------------------------")
        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, Data_set_name),
            use_folds=(fold,),
            checkpoint_name= checkpoint ,
        )
        
        checkpoint_output_folder = outdir  + '/' + str(checkpoint).replace('.pth','')
        if not os.path.exists(checkpoint_output_folder):
            os.makedirs(checkpoint_output_folder)
            

        #get a list of the predicted files from the first folder - all predictions are in the !!!!results folder!!!!
        # pred_list = [x.replace('.png',"") for x in os.listdir(indir) if x.endswith('.png')]
        pred_list = [x for x in os.listdir(indir) if x.endswith('.png')]
        output_folder_list = [join(checkpoint_output_folder, x.replace("_0000","").replace(".png","")) for x in pred_list]
        # print([[join(indir, x)] for x in pred_list])


        predictor.predict_from_files(
            [[join(indir, x)] for x in pred_list],
            output_folder_list,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )



def predict_from_folder_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name or id of the dataset')
    parser.add_argument('--fold', help='the fold to use for prediction')
    parser.add_argument('--input_folder', help='path to the folder with the images to predict')
    parser.add_argument('--output_folder', help='path to the folder where the predictions will be saved')
    parser.add_argument('--rule', help='late or sparse or both - which checkpoints to predict')
    args = parser.parse_args()

    
    predict_from_folder(args.dataset, args.fold, args.input_folder, args.output_folder, args.rule)

if __name__ == '__main__':
    predict_from_folder_entry()

