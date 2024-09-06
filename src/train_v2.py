from __future__ import annotations

import datetime
import os

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.autograd import Variable
from torch.cuda import amp
from torch.distributed import destroy_process_group
from torch.distributed import get_rank
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
from src.dataset import CarSegmentationData
from src.evaluation.valid import valid
from src.loss import ClsLoss
from src.loss import PixLoss
from src.models.birefnet import BiRefNet
from src.utils import AverageMeter
from src.utils import check_state_dict
from src.utils import Logger
from src.utils import set_seed


# Init model
def prepare_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    config,
    to_be_distributed=False,
    is_train=True,
):
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=min(config.num_workers, batch_size),
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset),
            drop_last=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=min(config.num_workers, batch_size, 0),
            pin_memory=True,
            shuffle=is_train,
            drop_last=True,
        )


def init_data_loaders(to_be_distributed, args, config):
    # Prepare dataset
    train_loader = prepare_dataloader(
        CarSegmentationData(datasets=args.trainset, config=config, is_train=True),
        batch_size=config.batch_size,
        config=config,
        to_be_distributed=to_be_distributed,
        is_train=True,
    )
    print(
        len(train_loader),
        "batches of train dataloader {} have been created.".format(args.trainset),
    )
    test_loaders = {}
    for testset in args.testsets:
        _data_loader_test = prepare_dataloader(
            CarSegmentationData(datasets=testset, config=config, is_train=False),
            batch_size=config.batch_size_valid,
            config=config,
            is_train=False,
        )
        print(
            len(_data_loader_test),
            "batches of valid dataloader {} have been created.".format(testset),
        )
        test_loaders[testset] = _data_loader_test
    return train_loader, test_loaders


def init_models_optimizers(epochs, to_be_distributed, logger, args, config, device):
    bb_pretrained = True if args.resume == "" else False
    model = BiRefNet(bb_pretrained=bb_pretrained)

    if args.resume != "":
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location="cpu")
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip(".pth").split("epoch_")[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if to_be_distributed:
        model = model.to(device)
        model = DDP(model, device_ids=[device])
    else:
        model = model.to(device)
    if config.compile:
        model = torch.compile(
            model, mode=["default", "reduce-overhead", "max-autotune"][0]
        )
    if config.precisionHigh:
        torch.set_float32_matmul_precision("high")

    # Setting optimizer
    if config.optimizer == "AdamW":
        optimizer = optim.AdamW(
            params=model.parameters(), lr=config.lr, weight_decay=1e-2
        )
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs
        ],
        gamma=config.lr_decay_rate,
    )
    logger.info("Optimizer details:")
    logger.info(optimizer)
    logger.info("Scheduler details:")
    logger.info(lr_scheduler)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self,
        data_loaders,
        model_opt_lrsch,
        trainer_scaler,
        trainer_config,
        trainer_args,
        trainer_to_be_distributed,
        trainer_device,
        logger,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader, self.test_loaders = data_loaders
        self.scaler = trainer_scaler
        self.config = trainer_config
        self.args = trainer_args
        self.to_be_distributed = trainer_to_be_distributed
        self.device = trainer_device
        self.logger = logger

        if trainer_config.out_ref:
            self.criterion_gdt = (
                nn.BCELoss() if not trainer_config.use_fp16 else nn.BCEWithLogitsLoss()
            )

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()

        # Others
        self.loss_log = AverageMeter()
        if trainer_config.lambda_adv_g:
            (
                self.optimizer_d,
                self.lr_scheduler_d,
                self.disc,
                self.adv_criterion,
            ) = self._load_adv_components()
            self.disc_update_for_odd = 0

    def _load_adv_components(self):
        # AIL
        from loss import Discriminator

        disc = Discriminator(channels=3, img_size=self.config.size)
        if self.to_be_distributed:
            disc = disc.to(self.device)
            disc = DDP(disc, device_ids=[self.device], broadcast_buffers=False)
        else:
            disc = disc.to(self.device)
        if self.config.compile:
            disc = torch.compile(
                disc, mode=["default", "reduce-overhead", "max-autotune"][0]
            )
        adv_criterion = (
            nn.BCELoss() if not self.config.use_fp16 else nn.BCEWithLogitsLoss()
        )
        if self.config.optimizer == "AdamW":
            optimizer_d = optim.AdamW(
                params=disc.parameters(), lr=self.config.lr, weight_decay=1e-2
            )
        elif self.config.optimizer == "Adam":
            optimizer_d = optim.Adam(
                params=disc.parameters(), lr=self.config.lr, weight_decay=0
            )
        lr_scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_d,
            milestones=[
                lde if lde > 0 else self.args.epochs + lde + 1
                for lde in self.config.lr_decay_epochs
            ],
            gamma=self.config.lr_decay_rate,
        )
        return optimizer_d, lr_scheduler_d, disc, adv_criterion

    def _train_batch(self, batch):
        inputs = batch[0].to(self.device)
        gts = batch[1].to(self.device)
        class_labels = batch[2].to(self.device)
        if self.config.use_fp16:
            with amp.autocast(enabled=self.config.use_fp16):
                scaled_preds, class_preds_lst = self.model(inputs)
                if self.config.out_ref:
                    (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
                    loss_gdt = 0.0
                    for _idx, (_gdt_pred, _gdt_label) in enumerate(
                        zip(outs_gdt_pred, outs_gdt_label)
                    ):
                        _gdt_pred = nn.functional.interpolate(
                            _gdt_pred,
                            size=_gdt_label.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )  # .sigmoid()
                        # _gdt_label = _gdt_label.sigmoid()
                        loss_gdt = (
                            self.criterion_gdt(_gdt_pred, _gdt_label)
                            if _idx == 0
                            else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
                        )
                    # self.loss_dict['loss_gdt'] = loss_gdt.item()
                if None in class_preds_lst:
                    loss_cls = 0.0
                else:
                    loss_cls = self.cls_loss(class_preds_lst, class_labels) * 1.0
                    self.loss_dict["loss_cls"] = loss_cls.item()

                # Loss
                loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * 1.0
                self.loss_dict["loss_pix"] = loss_pix.item()
                # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
                loss = loss_pix + loss_cls
                if self.config.out_ref:
                    loss = loss + loss_gdt * 1.0

                if self.config.lambda_adv_g:
                    # gen
                    valid = Variable(
                        torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(1.0),
                        requires_grad=False,
                    ).to(self.device)
                    adv_loss_g = (
                        self.adv_criterion(self.disc(scaled_preds[-1] * inputs), valid)
                        * self.config.lambda_adv_g
                    )
                    loss += adv_loss_g
                    self.loss_dict["loss_adv"] = adv_loss_g.item()
                    self.disc_update_for_odd += 1
            # self.loss_log.update(loss.item(), inputs.size(0))
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config.lambda_adv_g and self.disc_update_for_odd % 2 == 0:
                # disc
                fake = Variable(
                    torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(0.0),
                    requires_grad=False,
                ).to(self.device)
                adv_loss_real = self.adv_criterion(self.disc(gts * inputs), valid)
                adv_loss_fake = self.adv_criterion(
                    self.disc(scaled_preds[-1].detach() * inputs.detach()), fake
                )
                adv_loss_d = (
                    (adv_loss_real + adv_loss_fake) / 2 * self.config.lambda_adv_d
                )
                self.loss_dict["loss_adv_d"] = adv_loss_d.item()
                # self.optimizer_d.zero_grad()
                # adv_loss_d.backward()
                # self.optimizer_d.step()
                self.optimizer_d.zero_grad()
                self.scaler.scale(adv_loss_d).backward()
                self.scaler.step(self.optimizer_d)
                self.scaler.update()
        else:
            scaled_preds, class_preds_lst = self.model(inputs)
            if self.config.out_ref:
                (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
                for _idx, (_gdt_pred, _gdt_label) in enumerate(
                    zip(outs_gdt_pred, outs_gdt_label)
                ):
                    _gdt_pred = nn.functional.interpolate(
                        _gdt_pred,
                        size=_gdt_label.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    ).sigmoid()
                    _gdt_label = _gdt_label.sigmoid()
                    loss_gdt = (
                        self.criterion_gdt(_gdt_pred, _gdt_label)
                        if _idx == 0
                        else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
                    )
                # self.loss_dict['loss_gdt'] = loss_gdt.item()
            if None in class_preds_lst:
                loss_cls = 0.0
            else:
                loss_cls = self.cls_loss(class_preds_lst, class_labels) * 1.0
                self.loss_dict["loss_cls"] = loss_cls.item()

            # Loss
            loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * 1.0
            self.loss_dict["loss_pix"] = loss_pix.item()
            # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
            loss = loss_pix + loss_cls
            if self.config.out_ref:
                loss = loss + loss_gdt * 1.0

            if self.config.lambda_adv_g:
                # gen
                valid = Variable(
                    torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(1.0),
                    requires_grad=False,
                ).to(self.device)
                adv_loss_g = (
                    self.adv_criterion(self.disc(scaled_preds[-1] * inputs), valid)
                    * self.config.lambda_adv_g
                )
                loss += adv_loss_g
                self.loss_dict["loss_adv"] = adv_loss_g.item()
                self.disc_update_for_odd += 1
            self.loss_log.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.config.lambda_adv_g and self.disc_update_for_odd % 2 == 0:
                # disc
                fake = Variable(
                    torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(0.0),
                    requires_grad=False,
                ).to(self.device)
                adv_loss_real = self.adv_criterion(self.disc(gts * inputs), valid)
                adv_loss_fake = self.adv_criterion(
                    self.disc(scaled_preds[-1].detach() * inputs.detach()), fake
                )
                adv_loss_d = (
                    (adv_loss_real + adv_loss_fake) / 2 * self.config.lambda_adv_d
                )
                self.loss_dict["loss_adv_d"] = adv_loss_d.item()
                self.optimizer_d.zero_grad()
                adv_loss_d.backward()
                self.optimizer_d.step()

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}

        if epoch > self.args.epochs + self.config.IoU_finetune_last_epochs:
            self.pix_loss.lambdas_pix_last["bce"] *= 0
            self.pix_loss.lambdas_pix_last["ssim"] *= 1
            self.pix_loss.lambdas_pix_last["iou"] *= 0.5

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = "Epoch[{0}/{1}] Iter[{2}/{3}].".format(
                    epoch, self.args.epochs, batch_idx, len(self.train_loader)
                )
                info_loss = "Training Losses"
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ", {}: {:.3f}".format(loss_name, loss_value)
                self.logger.info(" ".join((info_progress, info_loss)))

        info_loss = "@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  ".format(
            epoch, self.args.epochs, loss=self.loss_log
        )

        for loss_name, loss_value in self.loss_dict.items():
            wandb.log({loss_name: loss_value, "epoch": epoch})

        self.lr_scheduler.step()
        if self.config.lambda_adv_g:
            self.lr_scheduler_d.step()

        return self.loss_log.avg

    def validate_model(self, epoch):
        num_image_testset_all = {
            "DIS-VD": 470,
            "DIS-TE1": 500,
            "DIS-TE2": 500,
            "DIS-TE3": 500,
            "DIS-TE4": 500,
        }
        num_image_testset = {}
        for testset in self.args.testsets:
            if "DIS-TE" in testset:
                num_image_testset[testset] = num_image_testset_all[testset]
        weighted_scores = {
            "f_max": 0,
            "f_mean": 0,
            "f_wfm": 0,
            "sm": 0,
            "e_max": 0,
            "e_mean": 0,
            "mae": 0,
        }
        len_all_data_loaders = 0
        self.model.epoch = epoch
        for testset, data_loader_test in self.test_loaders.items():
            print("Validating {}...".format(testset))
            performance_dict = valid(
                self.model,
                data_loader_test,
                pred_dir=".",
                method=(
                    self.args.ckpt_dir.split("/")[-1]
                    if self.args.ckpt_dir.split("/")[-1].strip(".").strip("/")
                    else "tmp_val"
                ),
                testset=testset,
                only_S_MAE=self.config.only_S_MAE,
                device=self.device,
            )
            print("Test set: {}:".format(testset))
            if self.config.only_S_MAE:
                print(
                    "Smeasure: {:.4f}, MAE: {:.4f}".format(
                        performance_dict["sm"], performance_dict["mae"]
                    )
                )
            else:
                print(
                    "Fmax: {:.4f}, Fwfm: {:.4f}, Smeasure: {:.4f}, Emean: {:.4f}, MAE: {:.4f}".format(
                        performance_dict["f_max"],
                        performance_dict["f_wfm"],
                        performance_dict["sm"],
                        performance_dict["e_mean"],
                        performance_dict["mae"],
                    )
                )
            if "-TE" in testset:
                for metric in (
                    ["sm", "mae"]
                    if self.config.only_S_MAE
                    else ["f_max", "f_mean", "f_wfm", "sm", "e_max", "e_mean", "mae"]
                ):
                    weighted_scores[metric] += performance_dict[metric] * len(
                        data_loader_test
                    )
                len_all_data_loaders += len(data_loader_test)
        print("Weighted Scores:")
        for metric, score in weighted_scores.items():
            if score:
                print("\t{}: {:.4f}.".format(metric, score / len_all_data_loaders))
                wandb.log({"val_score": metric, "score": score / len_all_data_loaders})


@hydra.main(version_base="1.3", config_path="../configs", config_name="default.yaml")
def main(cfg: DictConfig):
    config = hydra.utils.instantiate(cfg.get("config"))
    args = cfg.args

    if config.rand_seed:
        set_seed(config.rand_seed)

    # Half Precision
    _scaler = amp.GradScaler(enabled=config.use_fp16)

    to_be_distributed = args.dist
    if to_be_distributed:
        init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=3600 * 10)
        )
        device = int(os.environ["LOCAL_RANK"])
    else:
        device = config.device

    epoch_st = 1
    # make dir for ckpt
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Init log file
    logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
    logger_loss_idx = 1

    # log model and optimizer params
    # logger.info("Model details:"); logger.info(model)
    logger.info(
        "datasets: load_all={}, compile={}.".format(config.load_all, config.compile)
    )
    logger.info("Other hyperparameters:")
    logger.info(args)
    print("batch size:", config.batch_size)

    if os.path.exists(
        os.path.join(config.data_root_dir, args.testsets.strip("+").split("+")[0])
    ):
        args.testsets = args.testsets.strip("+").split("+")
    else:
        args.testsets = []

    wandb.init(
        # set the wandb project where this run will be logged
        project="car-segmentation-birefnet",
        name=args.experiment_name,
        config={
            "training_set": args.trainset,
            "test_set": args.testsets,
            "epochs": args.epochs,
            "resume_checkpoint": args.resume,
            "ckpt_dir": args.ckpt_dir,
        },
    )

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed, args, config),
        model_opt_lrsch=init_models_optimizers(
            args.epochs, to_be_distributed, logger, args, config, device
        ),
        trainer_scaler=_scaler,
        trainer_config=config,
        trainer_args=args,
        trainer_to_be_distributed=to_be_distributed,
        trainer_device=device,
        logger=logger,
    )

    for epoch in range(epoch_st, args.epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        # Save checkpoint
        # DDP
        if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
            torch.save(
                (
                    trainer.model.module.state_dict()
                    if to_be_distributed
                    else trainer.model.state_dict()
                ),
                os.path.join(args.ckpt_dir, "epoch_{}.pth".format(epoch)),
            )
        if (
            config.val_step
            and epoch >= args.epochs - config.save_last
            and (args.epochs - epoch) % config.val_step == 0
        ):
            if to_be_distributed:
                if get_rank() == 0:
                    print("Validating at rank-{}...".format(get_rank()))
                    trainer.validate_model(epoch)
            else:
                trainer.validate_model(epoch)

    if to_be_distributed:
        destroy_process_group()


if __name__ == "__main__":
    main()