from typing import Tuple

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.models.birefnet import BiRefNet
from src.utils import check_state_dict


class InferenceHandler:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.device = device
        self.model = self._prepare_model(checkpoint_path)

    def _prepare_model(self, checkpoint_path: str):
        model = BiRefNet(bb_pretrained=False)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        return model

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""

        image = Image.fromarray(image).convert("RGB")
        input_images = self.transform_image(image).unsqueeze(0)

        return input_images.to(self.device)

    def _postprocess_output(
        self, scaled_preds: torch.Tensor, original_image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Postprocess the output of the model"""

        mask = (
            nn.functional.interpolate(
                scaled_preds[0].unsqueeze(0),
                size=original_image_size,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        pred = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        return pred, mask

    def process(self, image: np.ndarray):
        """Main function doing inference on the image"""

        original_image = image.copy()
        original_image_size = image.shape[:2]  # only get h, w

        preprocessed_image = self._preprocess_image(image)

        with torch.no_grad():
            # Get the prediction
            ## Normally, the model will return multi-scale segmentation masks
            scaled_preds = self.model(preprocessed_image)[-1].sigmoid()

        # Postprocess the output
        pred, mask = self._postprocess_output(scaled_preds, original_image_size)

        # Multiply the prediction with the original image
        return (pred * original_image).astype(np.uint8), mask

    @staticmethod
    def _preprocess_bbox(bbox: list, image_size: Tuple[int, int], padding: int = 0):
        h, w = image_size
        x1, y1, x2, y2 = bbox

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return (x1, y1, x2, y2)

    def process_with_bbox_guided(self, image: np.ndarray, bbox: list):
        """Run inference with bbox-guided, only predict the mask inside the bbox"""
        origin_h, origin_w = image.shape[:2]
        bbox = self._preprocess_bbox(bbox, (origin_h, origin_w))

        image_crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        # transform cropped image
        preprocessed_image = self._preprocess_image(image_crop.copy())

        with torch.no_grad():
            # Get the prediction
            ## Normally, the model will return multi-scale segmentation masks
            scaled_preds = self.model(preprocessed_image)[-1].sigmoid()

        # Postprocess the output
        pred, mask = self._postprocess_output(
            scaled_preds, (bbox[3] - bbox[1], bbox[2] - bbox[0])
        )

        # Multiply the prediction with the crop image
        pred = (pred * image_crop).astype(np.uint8)

        # Create the output image
        image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = pred

        # create the mask
        mask_full = np.zeros((origin_h, origin_w), dtype=np.float32)
        mask_full[bbox[1] : bbox[3], bbox[0] : bbox[2]] = mask

        return image, mask_full

    def save_output(
        self,
        image_name: str,
        original_image: np.ndarray,
        output_image: np.ndarray,
        mask: np.ndarray,
        mask_threshold: float = 0.5,
        output_path: str = "",
    ):
        """Save the output image and mask"""

        # apply color mask to the output image, apply color with alpha 0.5
        color_mask = np.zeros_like(original_image)
        color_mask[mask >= mask_threshold] = [0, 0, 255]
        original_image = cv2.addWeighted(original_image, 0.7, color_mask, 0.5, 0)

        # save the color image
        save_output_path = os.path.join(output_path, "pred_images")
        if not os.path.exists(save_output_path):
            os.makedirs(save_output_path)

        cv2.imwrite(os.path.join(save_output_path, image_name), original_image)

        # convert to white background image instead of black (client requirement)
        output_image[mask < mask_threshold] = [255, 255, 255]
        # save the output image
        save_bg_path = os.path.join(output_path, "white_background")
        if not os.path.exists(save_bg_path):
            os.makedirs(save_bg_path)

        cv2.imwrite(os.path.join(save_bg_path, image_name), output_image)

        # save mask
        save_mask_path = os.path.join(output_path, "pred_masks")
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)

        cv2.imwrite(
            os.path.join(save_mask_path, image_name),
            (mask * 255).astype(np.uint8),
        )


if __name__ == "__main__":
    import os

    test_img_path = "/data/prod/1001/G118341_14.jpg"
    bbox = None
    # bbox = [int(x) for x in bbox]

    img = cv2.imread(test_img_path)

    handler = InferenceHandler(
        checkpoint_path="checkpoints/BiRefNet-massive-epoch_240.pth"
    )

    # predict
    if bbox is not None:
        output_image, mask = handler.process_with_bbox_guided(img.copy(), bbox)
    else:
        output_image, mask = handler.process(img.copy())

    # save output
    handler.save_output(
        image_name="5X53325_4.jpg",
        original_image=img,
        output_image=output_image,
        mask=mask,
        output_path="output/test_handler",
    )
