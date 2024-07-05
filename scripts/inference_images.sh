#!/bin/bash

image_path=$1
checkpoint=$2
save_path=${3:-"outputs/test_inference"}

echo "Running inference on ${image_path} with model ${checkpoint} and saving to ${save_path}"

python src/inference_images.py --image-path ${image_path} \
                               --checkpoint ${checkpoint} \
                               --save-path ${save_path}
