#!/bin/bash

testsets=$1
model_checkpoint=$2
save_dir=${3:-"outputs"}

echo "Running inference on ${testsets} with model ${model_checkpoint} and saving to ${save_dir}"

# Inference
devices=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=${devices} python src/inference.py --ckpt ${model_checkpoint} \
                                                        --pred_root ${save_dir} \
                                                        --testsets ${testsets}

echo Inference finished at $(date)
