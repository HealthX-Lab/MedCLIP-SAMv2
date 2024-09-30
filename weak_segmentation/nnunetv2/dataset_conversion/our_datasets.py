import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
# from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes
from PIL import Image
import numpy as np
import cv2


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    # seg = io.imread(input_seg)
    # seg[seg == 255] = 1
    # print(seg.shape)
    seg = Image.open(input_seg)
    img = Image.open(input_image)

    # Convert to grayscale or RGB
    seg = seg.convert('L')
    img = img.convert('RGB')

    # img = img.point(lambda p: 1 if p == 255 else 0)
    seg = np.array(seg)
    img = np.array(img)
    
    seg = cv2.resize(seg, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    seg[seg > 0] = 1
    # image = io.imread(input_image)
    # image = image.sum(2)
    # mask = image == (3 * 255)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    # mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
    #                                                                      sizes[j] > min_component_size])
    # mask = binary_fill_holes(mask)
    # seg[mask] = 0
    io.imsave(output_seg, seg, check_contrast=False)
    io.imsave(output_image, img, check_contrast=False)
    # shutil.copy(input_image, output_image)


if __name__ == "__main__":
    # extracted archive from https:\\/www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = "MedCLIP-SAMv2_data/LungCTPseudo"

    dataset_name = 'Dataset008_LungCTPseudo'

    nnUNet_raw = "data/nnUNet_raw"

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    # train_source = join(source, 'training')
    # test_source = join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(source, 'train_masks'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'train_images', v),
                         join(source, 'train_masks', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )

        # test set
        valid_ids = subfiles(join(source, 'test_masks'), join=False, suffix='png')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'test_images', v),
                         join(source, 'test_masks', v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'foreground': 1},
                          num_train, '.png', dataset_name=dataset_name)
