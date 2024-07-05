testsets=$1 # name of the testset folder. Eg: test_01042024
experiment_name=$2 # name of the experiment to evaluate
devices=${3:-0} # gpu device, default always 0
save_dir=${4:-outputs}

# Inference
echo "Running inference on ${testsets} with checkpoint from ${experiment_name} and saving to ${save_dir}"
CUDA_VISIBLE_DEVICES=${devices} python src/inference.py --experiment_name ${experiment_name} \
                                                        --ckpt_folder ckpt/${experiment_name} \
                                                        --pred_root ${save_dir} \
                                                        --testsets ${testsets}

echo Inference finished at $(date)

# Evaluate
testsets=(`echo ${testsets} | tr ',' ' '`) && testsets=${testsets[@]}

echo "Running evaluation on ${testsets}"
for testset in ${testsets}; do
    echo "Evaluating ${testset}"
    python src/eval_car_segmentation.py --pred_root ${save_dir} \
                                        --testsets ${testset} \
                                        --experiment_name ${experiment_name} \
                                        --dataset_root datasets \
                                        --check_integrity True
done


echo Evaluation finished at $(date)
