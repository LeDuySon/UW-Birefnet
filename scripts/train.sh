#!/bin/bash
# Run script
# Settings of training & test for different tasks.
experiment_name="$1" # experiment name
pretrained_checkpoint=${2:-""} # pretrained checkpoint

echo "Experiment name: ${experiment_name}"

task=$(python3 src/config.py)
case "${task}" in
    "DIS5K") epochs=600 && val_last=100 && step=5 ;;
    "COD") epochs=150 && val_last=50 && step=5 ;;
    "HRSOD") epochs=150 && val_last=50 && step=5 ;;
    "DIS5K+HRSOD+HRS10K") epochs=250 && val_last=50 && step=5 ;;
    "P3M-10k") epochs=150 && val_last=50 && step=5 ;;
    "car-segmentation") epochs=150 && val_last=50 && step=5 ;;
esac

trainset=$3
testsets=$4     # Non-existing folder to skip.


# Train
devices=${CUDA_VISIBLE_DEVICES:-0}
nproc_per_node=$(echo ${devices%%,} | grep -o "," | wc -l)

to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)
if [ ${to_be_distributed} == "True" ]
then
    # Adapt the nproc_per_node by the number of GPUs. Give 8989 as the default value of master_port.
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --nproc_per_node $((nproc_per_node+1)) --master_port=${3:-8989} \
    src/train.py --ckpt_dir ckpt/${experiment_name} --epochs ${epochs} \
        --testsets ${testsets} \
        --dist ${to_be_distributed}
else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    python src/train.py --experiment_name ${experiment_name} \
                        --ckpt_dir ckpt/${experiment_name} \
                        --epochs ${epochs} \
                        --trainset ${trainset} \
                        --testsets ${testsets} \
                        --dist ${to_be_distributed} \
                        --resume ${pretrained_checkpoint}

fi

echo Training finished at $(date)
