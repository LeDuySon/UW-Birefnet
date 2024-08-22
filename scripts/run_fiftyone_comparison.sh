#!/bin/bash

# checkpoint can be a folder or a specific checkpoint file
checkpoints=$1
# dataset can be a folder contains images or a dataset name in datasets/ folder (test_01042024)
dataset=${2:-"test_01042024"}
# Save the result to the folder
save_dir=${3:-"outputs/fiftyone_comparison"}
# test or inference mode -> test mode: compare the current checkpoint with groundtruth dataset
# inference mode: Compare different checkpoint using the dataset or image of folder
mode=${4:-"test"}

echo "Running fiftyone comparison on ${checkpoints} with dataset ${dataset} and mode ${mode}"

# Function to check if the given path is a file or a folder
check_path() {
    local path="$1"
    local result=()

    if [ -f "$path" ]; then
        # If it's a file, just return the file path
        result+=("$path")
        echo "${result[@]}"
        return 0  # True
    elif [ -d "$path" ]; then
        # If it's a directory, return all .pth files in it
        local pth_files=($(find "$path" -maxdepth 1 -name "*.pth"))
        if [ ${#pth_files[@]} -eq 0 ]; then
            result+=("No .pth files found in the directory.")
            echo "${result[@]}"
            return 1  # False if no .pth files found
        else
            for file in "${pth_files[@]}"; do
                result+=("$file")
            done
            echo "${result[@]}"
            return 0  # True
        fi
    else
        result+=("$path does not exist or is not a regular file or directory.")
        echo "${result[@]}"
        return 1  # False
    fi
}

result=$(check_path "$checkpoints")
checkpoints_file=($result)

if [ $? -eq 0 ]; then
    echo -e "### Processing with checkpoints: \n$result"
else
    echo -e "$result"
    exit 1 # stop the script
fi

for checkpoint in "${checkpoints_file[@]}"; do
    echo "Processing with $checkpoint"

    if [ $mode == "test" ]; then
        bash scripts/inference_dataset.sh ${dataset} ${checkpoint} ${savedir}
    else
        bash scripts/inference_images.sh ${dataset} ${checkpoint} ${savedir}
    fi
done
