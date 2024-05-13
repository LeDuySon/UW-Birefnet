import os
from glob import glob
from tqdm import tqdm
import cv2
import torch
from torch import nn
from models.birefnet import BiRefNet
from utils import save_tensor_img, check_state_dict , path_to_image
import requests
from PIL import Image
from io import BytesIO
import numpy as np

model = BiRefNet(bb_pretrained=False)
state_dict = torch.load("ckpt/car_segmentation_v1/epoch_50.pth", map_location='cpu')
state_dict = check_state_dict(state_dict)
model.load_state_dict(state_dict)
model.eval()
model = model.to('cuda')

from torchvision import transforms

transform_image = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def inference(image):
    image_copied = image.copy()
    
    image = Image.fromarray(image).convert('RGB')
    input = transform_image(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        scaled_preds = model(input)[-1].sigmoid()
        
    res = nn.functional.interpolate(
        scaled_preds[0].unsqueeze(0),
                    size=image_copied.shape[:2],
                    mode='bilinear',
                    align_corners=True
                ).squeeze().cpu().numpy()
    
    pred = np.repeat(np.expand_dims(res, axis=-1), 3, axis=-1)
    return (pred * image).astype(np.uint8), res

if __name__ == "__main__":
    dataset = "test_01042024_v1"
    data_path = f"../data/images/filtered_data/{dataset}"
    
    image_paths = glob(os.path.join(data_path, "*.jpg"))
    print(len(image_paths))
    
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        output_image, mask = inference(image)
        # convert to white bg
        output_image[mask < 0.5] = [255, 255, 255]
        
        save_path = os.path.join("../data/output", dataset, "car_segmentation_v1_epoch_50.pth")
        save_bg_path = os.path.join(save_path, "bg")
        if(not os.path.exists(save_bg_path)):
            os.makedirs(save_bg_path)
            
        cv2.imwrite(os.path.join(save_bg_path, os.path.basename(image_path)), output_image)
        
        save_mask_path = os.path.join(save_path, "mask")
        if(not os.path.exists(save_mask_path)):
            os.makedirs(save_mask_path)
        
        cv2.imwrite(os.path.join(save_mask_path, os.path.basename(image_path)), (mask * 255).astype(np.uint8))
        