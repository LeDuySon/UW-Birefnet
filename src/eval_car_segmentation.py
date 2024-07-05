import os
import argparse
from glob import glob
import prettytable as pt

from src.evaluation.evaluate import evaluator
from src.config import Config


config = Config()


def do_eval(args):
    # evaluation for whole dataset
    # dataset first in evaluation
    for _data_name in args.testsets.split("+"):
        pred_data_dir = sorted(
            glob(
                os.path.join(
                    args.pred_root, args.experiment_name, args.model_lst[0], _data_name
                )
            )
        )
        if not pred_data_dir:
            print("Skip dataset {}.".format(_data_name))
            continue
        gt_src = os.path.join(args.dataset_root, _data_name, "gt", "masks")
        gt_paths = sorted(glob(os.path.join(gt_src, "*")), key=lambda x: os.path.basename(x))
        print("#" * 20, _data_name, "#" * 20)

        filename = os.path.join(args.save_dir, "{}_eval.txt".format(_data_name))

        tb = pt.PrettyTable()
        tb.vertical_char = "&"
        if config.task == "DIS5K":
            tb.field_names = [
                "Dataset",
                "Method",
                "maxFm",
                "wFmeasure",
                "MAE",
                "Smeasure",
                "meanEm",
                "HCE",
                "maxEm",
                "meanFm",
                "adpEm",
                "adpFm",
            ]
        elif config.task == "COD":
            tb.field_names = [
                "Dataset",
                "Method",
                "Smeasure",
                "wFmeasure",
                "meanFm",
                "meanEm",
                "maxEm",
                "MAE",
                "maxFm",
                "adpEm",
                "adpFm",
                "HCE",
            ]
        elif config.task == "HRSOD":
            tb.field_names = [
                "Dataset",
                "Method",
                "Smeasure",
                "maxFm",
                "meanEm",
                "MAE",
                "maxEm",
                "meanFm",
                "wFmeasure",
                "adpEm",
                "adpFm",
                "HCE",
            ]
        elif config.task == "DIS5K+HRSOD+HRS10K":
            tb.field_names = [
                "Dataset",
                "Method",
                "maxFm",
                "wFmeasure",
                "MAE",
                "Smeasure",
                "meanEm",
                "HCE",
                "maxEm",
                "meanFm",
                "adpEm",
                "adpFm",
            ]
        elif config.task == "P3M-10k":
            tb.field_names = [
                "Dataset",
                "Method",
                "Smeasure",
                "maxFm",
                "meanEm",
                "MAE",
                "maxEm",
                "meanFm",
                "wFmeasure",
                "adpEm",
                "adpFm",
                "HCE",
            ]
        else:
            tb.field_names = [
                "Dataset",
                "Method",
                "Smeasure",
                "MAE",
                "maxEm",
                "meanEm",
                "maxFm",
                "meanFm",
                "wFmeasure",
                "adpEm",
                "adpFm",
                "HCE",
            ]
            
        for _model_name in args.model_lst[:]:
            print("\t", "Evaluating model: {}...".format(_model_name))
            pred_src = os.path.join(
                    args.pred_root, args.experiment_name, _model_name, _data_name
                )
            pred_paths = sorted(glob(os.path.join(pred_src, "*")), key=lambda x: os.path.basename(x))
            
            # print(pred_paths[:1], gt_paths[:1])
            em, sm, fm, mae, wfm, hce = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=args.metrics.split("+"),
                verbose=config.verbose_eval,
            )
            if config.task == "DIS5K":
                scores = [
                    fm["curve"].max().round(3),
                    wfm.round(3),
                    mae.round(3),
                    sm.round(3),
                    em["curve"].mean().round(3),
                    int(hce.round()),
                    em["curve"].max().round(3),
                    fm["curve"].mean().round(3),
                    em["adp"].round(3),
                    fm["adp"].round(3),
                ]
            elif config.task == "COD":
                scores = [
                    sm.round(3),
                    wfm.round(3),
                    fm["curve"].mean().round(3),
                    em["curve"].mean().round(3),
                    em["curve"].max().round(3),
                    mae.round(3),
                    fm["curve"].max().round(3),
                    em["adp"].round(3),
                    fm["adp"].round(3),
                    int(hce.round()),
                ]
            elif config.task == "HRSOD":
                scores = [
                    sm.round(3),
                    fm["curve"].max().round(3),
                    em["curve"].mean().round(3),
                    mae.round(3),
                    em["curve"].max().round(3),
                    fm["curve"].mean().round(3),
                    wfm.round(3),
                    em["adp"].round(3),
                    fm["adp"].round(3),
                    int(hce.round()),
                ]
            elif config.task == "DIS5K+HRSOD+HRS10K":
                scores = [
                    fm["curve"].max().round(3),
                    wfm.round(3),
                    mae.round(3),
                    sm.round(3),
                    em["curve"].mean().round(3),
                    int(hce.round()),
                    em["curve"].max().round(3),
                    fm["curve"].mean().round(3),
                    em["adp"].round(3),
                    fm["adp"].round(3),
                ]
            elif config.task == "P3M-10k":
                scores = [
                    sm.round(3),
                    fm["curve"].max().round(3),
                    em["curve"].mean().round(3),
                    mae.round(3),
                    em["curve"].max().round(3),
                    fm["curve"].mean().round(3),
                    wfm.round(3),
                    em["adp"].round(3),
                    fm["adp"].round(3),
                    int(hce.round()),
                ]
            else:
                scores = [
                    sm.round(3),
                    mae.round(3),
                    em["curve"].max().round(3),
                    em["curve"].mean().round(3),
                    fm["curve"].max().round(3),
                    fm["curve"].mean().round(3),
                    wfm.round(3),
                    em["adp"].round(3),
                    fm["adp"].round(3),
                    int(hce.round()),
                ]

            for idx_score, score in enumerate(scores):
                scores[idx_score] = (
                    "." + format(score, ".3f").split(".")[-1]
                    if score <= 1
                    else format(score, "<4")
                )
            records = [_data_name, _model_name] + scores
            tb.add_row(records)
            # Write results after every check.
            with open(filename, "w+") as file_to_write:
                file_to_write.write(str(tb) + "\n")
        print(tb)


if __name__ == "__main__":
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsets", type=str, help="test dataset. Eg: test_01042024")
    parser.add_argument(
        "--pred_root", type=str, help="prediction root", default="./e_preds"
    )
    parser.add_argument("--experiment_name", type=str, help="training experiment name")
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="path to the folder contain all datasets",
        default="datasets",
    )
    parser.add_argument(
        "--save_dir", type=str, help="candidate competitors", default="e_results"
    )
    parser.add_argument(
        "--check_integrity",
        type=bool,
        help="whether to check the file integrity",
        default=False,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="candidate competitors",
        default="+".join(
            ["S", "MAE", "E", "F", "WF", "HCE"][: 100 if "DIS5K" in config.task else -1]
        ),
    )
    args = parser.parse_args()

    args.save_dir = os.path.join(args.pred_root, args.experiment_name, "eval_results")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(">>> save_dir: ", args.save_dir)

    # get all models
    model_lst_path = os.listdir(os.path.join(args.pred_root, args.experiment_name))
    model_lst_path = [m for m in model_lst_path if m != "eval_results"]
    try:
        args.model_lst = [
            m
            for m in sorted(
                model_lst_path, key=lambda x: int(x.split("epoch_")[-1]), reverse=True
            )
            if int(m.split("epoch_")[-1]) % 1 == 0
        ]
    except:
        args.model_lst = [m for m in sorted(model_lst_path)]
        
    print(">>> model_lst: ", args.model_lst)

    # check the integrity of each candidates
    if args.check_integrity:
        for _data_name in args.testsets.split("+"):
            for _model_name in args.model_lst:
                gt_pth = os.path.join(args.dataset_root, _data_name, "gt", "masks")
                pred_pth = os.path.join(
                    args.pred_root, args.experiment_name, _model_name, _data_name
                )

                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(
                        len(sorted(os.listdir(gt_pth))),
                        len(sorted(os.listdir(pred_pth))),
                    )
                    print(
                        "The {} Dataset of {} Model is not matching to the ground-truth".format(
                            _data_name, _model_name
                        )
                    )
    else:
        print(">>> skip check the integrity of each candidates")

    # start engine
    do_eval(args)
