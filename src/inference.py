from __future__ import annotations

import argparse
import os
from glob import glob

import cv2
import torch
from tqdm import tqdm

from src.config import Config
from src.dataset import CarSegmentationData
from src.dataset import MyData
from src.models.birefnet import BiRefNet
from src.utils import check_state_dict
from src.utils import save_tensor_img


config = Config()


def inference(model, data_loader_test, pred_root, method, testset, device=0):
    model_training = model.training
    if model_training:
        model.eval()
    for batch in (
        tqdm(data_loader_test, total=len(data_loader_test))
        if 1 or config.verbose_eval
        else data_loader_test
    ):
        inputs = batch[0].to(device)
        # gts = batch[1].to(device)
        label_paths = batch[-1]
        with torch.no_grad():
            scaled_preds = model(inputs)[-1].sigmoid()

        os.makedirs(os.path.join(pred_root, method, testset), exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            res = torch.nn.functional.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),
                size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[
                    :2
                ],
                mode="bilinear",
                align_corners=True,
            )
            save_tensor_img(
                res,
                os.path.join(
                    os.path.join(pred_root, method, testset),
                    label_paths[idx_sample].replace("\\", "/").split("/")[-1],
                ),
            )  # test set dir + file name
    if model_training:
        model.train()
    return None


def main(args):
    # Init model

    device = config.device
    if args.ckpt:
        print("Testing with model {}".format(args.ckpt))
    else:
        print("Testing with models in {}".format(args.ckpt_folder))

    experiment_name = None
    if args.experiment_name != "":
        experiment_name = args.experiment_name
    else:
        experiment_name = "single_checkpoint"

    if config.model == "BiRefNet":
        model = BiRefNet(bb_pretrained=False)
    weights_lst = sorted(
        glob(os.path.join(args.ckpt_folder, "*.pth")) if not args.ckpt else [args.ckpt],
        key=lambda x: int(x.split("epoch_")[-1].split(".pth")[0]),
        reverse=True,
    )
    for testset in args.testsets.split("+"):
        print(">>>> Testset: {}...".format(testset))
        data_loader_test = torch.utils.data.DataLoader(
            dataset=CarSegmentationData(testset, config=config, is_train=False),
            batch_size=config.batch_size_valid,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        for weights in weights_lst:
            if int(weights.strip(".pth").split("epoch_")[-1]) % 1 != 0:
                continue
            print("\tInferencing {}...".format(weights))
            # model.load_state_dict(torch.load(weights, map_location='cpu'))
            state_dict = torch.load(weights, map_location="cpu")
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model = model.to(device)

            method = os.path.join(
                experiment_name,
                "--".join([w.split(".")[0] for w in weights.split(os.sep)[-2:]]),
            )
            print("Saving result to: ", os.path.join(args.pred_root, method, testset))
            inference(
                model,
                data_loader_test=data_loader_test,
                pred_root=args.pred_root,
                method=method,
                testset=testset,
                device=config.device,
            )


if __name__ == "__main__":
    # Parameter from command line
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ckpt", type=str, help="model folder")
    parser.add_argument(
        "--experiment_name",
        default="",
        type=str,
        help="The name of the training experiment",
    )
    parser.add_argument("--ckpt_folder", type=str, help="model folder")
    parser.add_argument(
        "--pred_root", default="outputs", type=str, help="Output folder"
    )
    parser.add_argument(
        "--testsets",
        type=str,
        help="Test all sets: , 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'",
    )

    args = parser.parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision("high")
    main(args)
