import numpy as np
import cv2
import os
import argparse
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import SimpleITK as sitk

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--mask-input",
    type=str,
    required=True,
    help="Path to either a single crf mask image or folder of crf mask images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--prompts",
    type=str,
    required=True,
    help="The type of prompts to use, in ['points', 'boxes', 'both']",
)

parser.add_argument(
    "--num-points",
    type=int,
    required=False,
    default=10,
    help="Number of points when using point prompts, default is 8",
)

parser.add_argument(
    "--negative",
    action="store_true",
    help="Whether to sample points in the background. Default is False.",
)

parser.add_argument(
    "--neg-num-points",
    type=int,
    required=False,
    default=10,
    help="Number of negative points when using negative mode, default is 20",
)

parser.add_argument(
    "--pos-margin",
    type=float,
    required=False,
    default=10,
    help="controls the sampling margin for the positive point prompts, default is 2, for large structures use above 15, but \
    for smaller objects use 2-5",
)

parser.add_argument(
    "--neg-margin",
    type=float,
    required=False,
    default=5,
    help="controls the sampling margin for the negative point prompts, default is 5",
)


parser.add_argument(
    "--multimask",
    action="store_true",
    help="Whether to output multimasks in SAM. Default is False.",
)

parser.add_argument(
    "--multicontour",
    action="store_true",
    help="Whether to output multiple bounding boxes for each contour. Default is False.",
)

parser.add_argument(
    "--voting",
    type=str,
    default="AVERAGE",
    help="['MRM','STAPLE','AVERAGE']",
)

parser.add_argument(
    "--plot",
    action="store_true",
    help="Whether to plot the points and boxes in the contours. Default is False.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

def write_mask_to_folder(mask , t_mask, path: str,num_contours) -> None:
    file = t_mask.split("/")[-1]
    filename = f"{file}"
    mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    mask = mask.astype(np.uint8)*255
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
    sizes = stats[:, cv2.CC_STAT_AREA]
    sorted_sizes = sorted(sizes[1:], reverse=True) 

    # Determine the top K sizes
    top_k_sizes = sorted_sizes[:num_contours]
    
    im_result = np.zeros_like(im_with_separated_blobs)
    
    for index_blob in range(1, nb_blobs):
        if sizes[index_blob] in top_k_sizes:
            im_result[im_with_separated_blobs == index_blob] = 255
    mask = im_result
    cv2.imwrite(os.path.join(path, filename), mask)

    return

def scoremap2bbox(scoremap, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = (scoremap * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        image=scoremap_image,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)
    
    num_contours = len(contours)

    if len(contours) == 0:
        return np.asarray([[0, 0, width, height]]), 1
    

    if not multi_contour_eval:
        # contours = [max(contours, key=cv2.contourArea)]
        contours = [np.concatenate(contours)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return estimated_boxes, contours,num_contours

def get_prompts(mask, args):
    # List to store bounding boxes and random points
        bounding_boxes = []
        all_random_points = []
        all_input_labels = []

        bounding_boxes, contours, num_contours = scoremap2bbox(mask, multi_contour_eval=args.multicontour)
        
        if(args.prompts == "boxes"):
            bounding_boxes = np.array(bounding_boxes)
            return np.zeros_like(bounding_boxes), np.zeros_like(bounding_boxes), bounding_boxes, num_contours
    
        pos_num_points = args.num_points  # number of positive random points to get per contour
        neg_num_points = args.neg_num_points  # number of negative random points to get per contour
        pos_random_points = []
        neg_random_points = []
        candidate_points = np.argwhere(mask.transpose(1,0) > 0)
        h,w = mask.shape
        random_index = np.random.choice(len(candidate_points), pos_num_points, replace=False)
        pos_random_points = candidate_points[random_index]

        if(args.negative):
            # Filter some points
            candidate_points = np.argwhere(mask.transpose(1,0) == 0)
            random_index = np.random.choice(len(candidate_points), neg_num_points, replace=False)
            neg_random_points = candidate_points[random_index]
            all_random_points = np.concatenate([pos_random_points,neg_random_points])
            all_input_labels = [1]*len(pos_random_points) + [0]*len(neg_random_points)
        
        elif(not args.negative):
            all_random_points = pos_random_points
            all_input_labels = [1]*len(pos_random_points)

        # Convert the lists to NumPy arrays
        all_random_points = np.array(all_random_points)
        all_input_labels = np.array(all_input_labels)
        bounding_boxes = np.array(bounding_boxes)

        return all_random_points, all_input_labels, bounding_boxes, num_contours
    
def get_final_mask(predictor,all_random_points, all_input_labels, 
                   bounding_boxes, image, args):
    
    input_boxes = torch.tensor(bounding_boxes, device=args.device) 
    input_points = torch.tensor(all_random_points, device=args.device)
    input_labels = torch.tensor(all_input_labels, device=args.device)
    input_points = input_points.repeat((len(bounding_boxes),1,1))
    input_labels = input_labels.repeat((len(bounding_boxes),1))

    predictor.set_image(image)

    if(args.prompts == "both"):
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]) 
        transformed_points = predictor.transform.apply_coords_torch(input_points, image.shape[:2]) 
        masks, scores, _ = predictor.predict_torch(
            point_coords=transformed_points,
            point_labels=input_labels,
            boxes=transformed_boxes,
            multimask_output=args.multimask
        )
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()

    elif(args.prompts == "points"):
        masks, scores, _ = predictor.predict(
            point_coords=all_random_points,
            point_labels=all_input_labels,
            multimask_output=args.multimask,
        )

    elif(args.prompts == "boxes"):
        input_boxes = torch.tensor(bounding_boxes, device=args.device)  
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])  
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=args.multimask
        )
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()

    if(args.multimask):
        if(args.voting == "MRM"):
            scores = np.expand_dims(scores, axis=-1)
            scores = np.expand_dims(scores, axis=-1)
            final_mask = (masks * scores).sum(axis=0)
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
        elif(args.voting == "STAPLE"):
            final_mask = []
            for i in range(masks.shape[0]):
                seg_sitk = sitk.GetImageFromArray(masks[i].astype(np.int16)) # STAPLE requires we cast into int16 arrays
                final_mask.append(seg_sitk)

            # Run STAPLE algorithm
            final_mask_img = sitk.STAPLE(final_mask, 1.0 ) # 1.0 specifies the foreground value

            # convert back to numpy array
            final_mask = sitk.GetArrayFromImage(final_mask_img)

        elif(args.voting == "AVERAGE"):
            final_mask = np.mean(masks, axis=0)
    else:
        final_mask = np.squeeze(masks).astype(float)

    if(final_mask.ndim == 3):
        final_mask = final_mask.sum(axis=0).clip(0, 1)

    return final_mask

def main(args: argparse.Namespace) -> None:

    print("Segmenting images using SAM...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)

    predictor = SamPredictor(sam)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    if not os.path.isdir(args.mask_input):
        targets_mask = [args.mask_input]
    else:
        targets_mask = [
            f for f in os.listdir(args.mask_input) if not os.path.isdir(os.path.join(args.mask_input, f))
        ]
        targets_mask = [os.path.join(args.mask_input, f) for f in targets_mask]

    os.makedirs(args.output, exist_ok=True)

    for t,t_mask in tqdm(zip(targets,targets_mask),total=len(targets)):

        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(t_mask, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        all_random_points, all_input_labels, bounding_boxes, num_contours = get_prompts(mask, args)


        final_mask = get_final_mask(predictor,all_random_points, all_input_labels, 
                                    bounding_boxes, image,args)

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]

        # Save the final mask
        write_mask_to_folder(final_mask, t_mask,args.output,num_contours)
    
    print("SAM Segmentation Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)