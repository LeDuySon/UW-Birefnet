import os
from glob import glob

import cv2
import numpy as np
import torch
import argparse
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchvision import transforms

from src.models.birefnet import BiRefNet
from src.utils import check_state_dict, path_to_image, save_tensor_img

transform_image = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model_from_checkpoint(checkpoint_path):
    model = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(
        checkpoint_path, map_location='cpu'
    )
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to('cuda')
    
    return model

def predict(model, image):
    copied_image = image.copy()
    
    image = Image.fromarray(image).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to('cuda')
        
    with torch.no_grad():
        scaled_preds = model(input_images)[-1].sigmoid()
                
    res = nn.functional.interpolate(
            scaled_preds[0].unsqueeze(0),
            size=copied_image.shape[:2],
            mode='bilinear',
            align_corners=True
        ).squeeze().cpu().numpy()
        
    pred = np.repeat(np.expand_dims(res, axis=-1), 3, axis=-1)
    return (pred * image).astype(np.uint8), res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image-path', required=True, type=str, help='Path to folder contains images')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint file')
    parser.add_argument('--save-path', required=True, type=str, help='Path to save output images')
    args = parser.parse_args()

    print("Loading images from {}".format(args.image_path))
    image_types = ["*.jpeg", "*.jpg", "*.png", "*.PNG", "*.JPG", "*.JPEG"]
    image_paths = []
    for image_type in image_types:
        image_paths += glob(os.path.join(args.image_path, image_type))
    print(f"Found {len(image_paths)} images")
    
    print("Loading model from {}".format(args.checkpoint))
    model = load_model_from_checkpoint(args.checkpoint)

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        output_image, mask = predict(model, image)
        # convert to white bg
        output_image[mask < 0.5] = [255, 255, 255]

        save_bg_path = os.path.join(args.save_path, "bg_removed")
        if not os.path.exists(save_bg_path):
            os.makedirs(save_bg_path)

        cv2.imwrite(
            os.path.join(save_bg_path, os.path.basename(image_path)), output_image
        )

        save_mask_path = os.path.join(args.save_path, "masks")
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)

        cv2.imwrite(
            os.path.join(save_mask_path, os.path.basename(image_path)),
            (mask * 255).astype(np.uint8),
        )
