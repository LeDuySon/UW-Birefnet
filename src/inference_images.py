from __future__ import annotations

import argparse
import os
from glob import glob

import cv2
from tqdm import tqdm

from src.inference_handler import InferenceHandler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--image-path", required=True, type=str, help="Path to folder contains images"
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--save-path", required=True, type=str, help="Path to save output images"
    )
    args = parser.parse_args()

    print("Loading images from {}".format(args.image_path))
    image_types = ["*.jpeg", "*.jpg", "*.png", "*.PNG", "*.JPG", "*.JPEG"]
    image_paths = []
    for image_type in image_types:
        image_paths += glob(os.path.join(args.image_path, image_type))
    print(f"Found {len(image_paths)} images")

    print("Loading model from {}".format(args.checkpoint))    
    inference_handler = InferenceHandler(args.checkpoint)

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        output_image, mask = inference_handler.process(image.copy())

        inference_handler.save_output(
            image_name=os.path.basename(image_path),
            original_image = image,
            output_image=output_image,
            mask=mask,
            output_path=args.save_path
        )
