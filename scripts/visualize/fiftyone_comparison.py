from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import cv2
import fiftyone as fo
import numpy as np
from tqdm import tqdm


def list_images(path):
    images = []
    suffix = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    for root, dirs, files in os.walk(path):
        for file in files:
            for s in suffix:
                if file.endswith(s):
                    images.append(os.path.join(root, file))
    return images


def get_groundtruth_mask_path(dataset):
    if dataset is None:
        return

    groundtruth_mask_path = os.path.join("datasets", dataset, "gt", "masks")
    return groundtruth_mask_path


def read_mask(mask_path, threshold: int = 125):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"File not found: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs",
        help="Path to the output folder contain predicted mask",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset")
    parser.add_argument(
        "--images-path", type=str, default=None, help="Path to the images folder"
    )
    args = parser.parse_args()

    if args.dataset is not None:
        image_path = os.path.join("datasets", args.dataset, "images")
    elif args.images_path is not None:
        image_path = args.images_path
    else:
        raise ValueError("Please provide the dataset or images path")

    # get original images
    images = list_images(image_path)
    images = sorted(images)

    # get groundtruth masks
    gt_mask_folder = get_groundtruth_mask_path(args.dataset)

    # set fiftyone dataset
    fo_dataset = fo.Dataset(name=args.dataset, overwrite=True)
    # Set default mask targets
    fo_dataset.default_mask_targets = {1: "car"}
    # Set mask targets for the `ground_truth` and `predictions` fields
    fo_dataset.mask_targets = {
        "ground_truth": {1: "car"},
        "predictions": {1: "car"},
    }

    checkpoints = os.listdir(args.output_path)
    checkpoint2masks = defaultdict(list)
    for checkpoint in checkpoints:
        checkpoint_output_path = os.path.join(args.output_path, checkpoint)
        # get predicted masks
        checkpoint_predicted_mask = list_images(checkpoint_output_path)

        # validate
        assert len(checkpoint_predicted_mask) == len(
            images
        ), f"Number of images and masks are not equal for {checkpoint_output_path}"

        checkpoint2masks[checkpoint] = sorted(checkpoint_predicted_mask)

    for idx, image in enumerate(tqdm(images)):
        print("Adding image: ", image)
        sample = fo.Sample(filepath=image)
        sample["img_path"] = image

        if gt_mask_folder is not None:
            print("Adding groundtruth mask...")
            gt_mask = read_mask(os.path.join(gt_mask_folder, os.path.basename(image)))
            sample["ground_truth"] = fo.Segmentation(mask=gt_mask)

        for checkpoint in checkpoints:
            checkpoint_masks = checkpoint2masks[checkpoint]
            mask = read_mask(checkpoint_masks[idx])
            sample[checkpoint.split(".")[0]] = fo.Segmentation(mask=mask)

        fo_dataset.add_sample(sample)

    # Launch the App
    session = fo.launch_app(dataset=fo_dataset, address="0.0.0.0", port=3000)

    # Blocks execution until the App is closed
    session.wait()
