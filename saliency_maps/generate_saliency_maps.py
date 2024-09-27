import warnings
warnings.filterwarnings('ignore')
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.clip_wrapper import ClipWrapper
from scripts.plot import visualize_vandt_heatmap
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import itertools
import torch
import random
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from PIL import Image
from scripts.methods import vision_heatmap_iba
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    dice_coefficient = (2.0 * intersection) / (mask1.sum() + mask2.sum())
    return dice_coefficient


def evaluate_on_sample(model, processor, tokenizer, text, image_paths, args):
    dice_scores = []
    for image_id in tqdm(image_paths):
        try:
            image = Image.open(f"{args.val_path}/{image_id}").convert('RGB')
        except:
            continue
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device)     
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
        vmap = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar,ensemble=args.ensemble, progbar=False)
        gt_path = args.val_path.replace("images","masks")
        gt_mask = np.array(Image.open(f"{gt_path}/{image_id}").convert("L"))
        vmap_resized = cv2.resize(np.array(vmap),(gt_mask.shape[1],gt_mask.shape[0]))
        cam_img = vmap_resized > 0.3
        dice_score = calculate_dice_coefficient(gt_mask.astype(bool),cam_img.astype(bool))
        dice_scores.append(dice_score)
    average_dice = np.mean(dice_scores)
    return average_dice

def hyper_opt(model,processor,tokenizer,text,args):

    print("Running Hyperparameter Optimization ...")

    # Define hyperparameter grid
    vbeta_list = [0.1, 0.5, 1.0, 1.5, 2.0]
    vvar_list = [0.1, 0.5, 1.0, 1.5, 2.0]
    layers_list = [7,8,9]


    hyperparameter_combinations = list(itertools.product(vbeta_list, vvar_list, layers_list))

    # Get all image IDs
    all_image_ids = sorted(os.listdir(args.val_path))

    results = []

    for combo in hyperparameter_combinations:
        vbeta, vvar, layer = combo
        args.vbeta = vbeta
        args.vvar = vvar
        args.vlayer = layer
        sample_dice_scores = []
        print(f"Evaluating combination: vbeta={vbeta}, vvar={vvar}, layer={layer}")
        for i in range(5):
            random.seed(i)
            sampled_images = random.sample(all_image_ids, 1)
            avg_dice = evaluate_on_sample(model, processor, tokenizer, text, sampled_images, args)
            sample_dice_scores.append(avg_dice)
            print(f"  Sample {i+1}: Average Dice Score = {avg_dice}")
        mean_dice = np.mean(sample_dice_scores)
        results.append({
            'vbeta': vbeta,
            'vvar': vvar,
            'vlayer': layer,
            'average_dice': mean_dice
        })
        print(f"Mean Dice Score for this combination: {mean_dice}\n")

    # Find the best hyperparameter combination
    results_df = pd.DataFrame(results)
    best_combo = results_df.loc[results_df['average_dice'].idxmax()]
    print("Best Hyperparameter Combination:")
    print(best_combo)
    return best_combo

def main(args):
    print("Loading models ...")
    if(args.model_name == "BiomedCLIP" and args.finetuned):
        model = AutoModel.from_pretrained("./saliency_maps/model", trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    elif(args.model_name == "BiomedCLIP" and args.finetuned == False):
        model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    elif(args.model_name == "CLIP" and args.finetuned == False):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    elif(args.model_name == "CLIP" and args.finetuned):
        model = AutoModel.from_pretrained("./model", trust_remote_code=True).to(device)
    text = str(input("Enter the text: "))
    if(args.hyper_opt):
        best_combo = hyper_opt(model,processor,tokenizer,text,args)
        args.vbeta = best_combo['vbeta']
        args.vvar = best_combo['vvar']
        args.vlayer = int(best_combo['vlayer'])
    print("Generating Saliency Maps ...")
    if(os.path.exists(args.output_path) == False):
        os.makedirs(args.output_path)
    for image_id in tqdm(sorted(os.listdir(args.input_path))):
        if(image_id in os.listdir(args.output_path)):
            continue
        try:
            image = Image.open(f"{args.input_path}/{image_id}").convert('RGB')
        except:
            print(f"Unable to load image at {image_id}", flush=True)
            continue
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device) # 3*224*224
        # Tokenize text

        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
        
        vmap = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar,ensemble=args.ensemble,progbar=False)
        img = np.array(image)
        vmap = cv2.resize(np.array(vmap),(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{args.output_path}/{image_id}", vmap*255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('M2IB argument parser')
    parser.add_argument('--input-path', required=True, default="data/input_images",type=str, help='path to the images')
    parser.add_argument('--output-path', required=True, default="saliency_map_outputs", type=str, help='path to the output')
    parser.add_argument('--val-path', type=str, default="data/val_images", 
                        help='path to the validation set for hyperparameter optimization')
    parser.add_argument('--vbeta', type=int, default=0.1)
    parser.add_argument('--vvar', type=int, default=1.0)
    parser.add_argument('--vlayer', type=int, default=7)
    parser.add_argument('--tbeta', type=int, default=0.3)
    parser.add_argument('--tvar', type=int, default=1)
    parser.add_argument('--tlayer', type=int, default=9)
    parser.add_argument('--model-name', type=str, default="BiomedCLIP", help="Which CLIP model to use")
    parser.add_argument('--finetuned',action='store_true',
                        help="Whether to use finetuned weights or not")
    parser.add_argument('--hyper-opt',action='store_true',
                        help="Whether to optimize hyperparameters or not")
    parser.add_argument('--ensemble',action='store_true',
                        help="Whether to use text ensemble or not")
    args = parser.parse_args()
    main(args)
    print("Saliency Map Generation Done!")
    