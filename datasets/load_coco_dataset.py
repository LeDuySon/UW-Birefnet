from pycocotools.coco import COCO

import numpy as np
import cv2
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load COCO dataset")
    parser.add_argument("--dataset-path", type=str, default="train_01042024_v1", help="Dataset to load")
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    
    save_gt_path = os.path.join(dataset_path, "gt")
    if(not os.path.exists(save_gt_path)):
        os.makedirs(save_gt_path)

    annotation_file_path = os.path.join(dataset_path, "annotations", "instances_default.json")
    coco = COCO(annotation_file_path)

    images_info = coco.imgs

    cat_ids = coco.getCatIds(catNms=["car", "background"])
    img_ids = coco.getImgIds()

    for idx in range(len(img_ids)):    
        img = coco.loadImgs(img_ids[idx])[0]

        ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        file_name = images_info[img_ids[idx]]['file_name']
        image_name = os.path.basename(file_name)
        
        mask = np.zeros((img['height'], img['width']))
        for i in range(len(anns)):
            if(anns[i]['category_id'] == cat_ids[1]): # idx 1 is background class
                mask[coco.annToMask(anns[i]) == 1] = 0
            else:
                mask = np.maximum(coco.annToMask(anns[i]), mask)

        save_mask_path = os.path.join(save_gt_path, "masks")
        if(not os.path.exists(save_mask_path)):
            os.makedirs(save_mask_path)
        
        cv2.imwrite(os.path.join(save_mask_path, f"{image_name}"), mask*255)
        
        # apply mask to image
        original_image_path = os.path.join(dataset_path, "images", file_name)
        print("Process: ", original_image_path)
        
        image = cv2.imread(original_image_path)
        image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        image[mask == 0] = 255
        
        # save image
        save_image_path = os.path.join(save_gt_path, "bg_removed")
        if(not os.path.exists(save_image_path)):
            os.makedirs(save_image_path)
            
        cv2.imwrite(os.path.join(save_image_path, f"{image_name}"), image)    