import argparse
import gc
import importlib
import os
import sys
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    Activationsd,
    AsDiscreted,
    LoadImage,
)
import json

from dataset import CustomDataset
from sklearn.metrics import roc_auc_score
import timm

def main(cfg):

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)
    set_seed(cfg.seed)
    # set dataset, dataloader
    df = pd.read_csv(cfg.data_df)

    val_df = df[df["fold"] == cfg.fold]
    train_df = df[df["fold"] != cfg.fold]

    train_dataset = CustomDataset(df=train_df, cfg=cfg, aug=cfg.train_transforms)
    val_dataset = CustomDataset(df=val_df, cfg=cfg, aug=cfg.val_transforms)
    print("train: ", len(train_dataset), " val: ", len(val_dataset))
    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    # set model
    model = timm.create_model(
        cfg.backbone,
        pretrained=True,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )
    # model = EfnNet()
    model = torch.nn.DataParallel(model)
    model.to(cfg.device)
    if cfg.weights is not None:
        model.load_state_dict(
            torch.load(os.path.join(f"{cfg.output_dir}/fold{cfg.fold}", cfg.weights))[
                "model"
            ]
        )
        print(f"weights from: {cfg.weights} are loaded.")

    # set optimizer, lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=int(len(train_dataset) / cfg.batch_size),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=cfg.lr_div,
        final_div_factor=cfg.lr_final_div,
    )
    # set loss
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([cfg.pos_weight])).to(cfg.device)

    # set other tools
    scaler = GradScaler()
    writer = SummaryWriter(str(cfg.output_dir + f"/fold{cfg.fold}/"))

    # train and val loop
    step = 0
    i = 0
    best_metric = 0.0
    optimizer.zero_grad()
    print("start from: ", best_metric)
    for epoch in range(cfg.epochs):
        print("EPOCH:", epoch)
        gc.collect()
        run_train(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            scaler=scaler,
            writer=writer,
            epoch=epoch,
            iteration=i,
            step=step,
            loss_function=loss_function,
        )

        val_metric = run_eval(
            model=model,
            val_dataloader=val_dataloader,
            cfg=cfg,
            writer=writer,
            epoch=epoch,
        )

        if val_metric > best_metric:
            print(f"SAVING CHECKPOINT: val_metric {best_metric:.5} -> {val_metric:.5}")
            best_metric = val_metric

            checkpoint = create_checkpoint(
                model,
                optimizer,
                epoch,
                scheduler=scheduler,
                scaler=scaler,
            )
            torch.save(
                checkpoint,
                f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best_metric.pth",
            )


def run_train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    cfg,
    scaler,
    writer,
    epoch,
    iteration,
    step,
    loss_function,
):
    model.train()
    losses = []
    progress_bar = tqdm(range(len(train_dataloader)))
    tr_it = iter(train_dataloader)

    all_outputs, all_labels = [], []

    for itr in progress_bar:
        batch = next(tr_it)
        inputs, labels = batch["image"].to(cfg.device), batch["label"].float().to(cfg.device)
        iteration += 1

        step += cfg.batch_size
        torch.set_grad_enabled(True)
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        losses.append(loss.item())

        outputs = list(torch.sigmoid(outputs).detach().cpu().numpy())
        labels = list(labels.detach().cpu().numpy())

        all_outputs.extend(outputs)
        all_labels.extend(labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        progress_bar.set_description(f"loss: {np.mean(losses):.2f} lr: {scheduler.get_last_lr()[0]:.6f}")
    score = pfbeta(all_labels, all_outputs, 1.0)[0]
    auc = roc_auc_score(all_labels, all_outputs)
    print("Train F1: ", score, "AUC: ", auc)


def run_eval(model, val_dataloader, cfg, writer, epoch):

    model.eval()
    torch.set_grad_enabled(False)

    progress_bar = tqdm(range(len(val_dataloader)))
    tr_it = iter(val_dataloader)

    all_labels = []
    all_outputs = []
    all_ids = []

    for itr in progress_bar:
        batch = next(tr_it)
        inputs, labels = batch["image"].to(cfg.device), batch["label"].float().to(cfg.device)
        ids = batch["prediction_id"]
        outputs = model(inputs)
        outputs = list(torch.sigmoid(outputs).detach().cpu().numpy()[:,0])
        labels = list(labels.detach().cpu().numpy()[:,0])

        all_outputs.extend(outputs)
        all_labels.extend(labels)
        all_ids.extend(ids)

    df_pred = pd.DataFrame.from_dict(all_ids)
    df_pred.columns = ["prediction_id"]
    df_pred["all_labels"] = all_labels
    df_pred["all_outputs"] = all_outputs
    df_pred = df_pred.groupby(["prediction_id"]).mean().reset_index()
    all_labels = df_pred["all_labels"]
    all_outputs = df_pred["all_outputs"]

    score = pfbeta(all_labels, all_outputs, 1.0)
    auc = roc_auc_score(all_labels, all_outputs)
    all_outputs = (np.array(all_outputs) > cfg.clf_threshold).astype(np.int8).tolist()
    try:
        bin_score = pfbeta(all_labels, all_outputs, 1.0)
    except:
        bin_score = 0.0
    print("Val F1: ", score, "Val Bin F1: ", bin_score, "AUC: ", auc)
    writer.add_scalar("F1", bin_score, epoch)

    return score


if __name__ == "__main__":

    sys.path.append("configs")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-c", "--config", default="cfg_clf_baseline", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=0, help="fold")
    parser.add_argument("-backbone", "--backbone", default="tf_efficientnetv2_s", help="backbone")
    parser.add_argument("-s", "--seed", type=int, default=20220421, help="seed")
    parser.add_argument("-w", "--weights", default=None, help="the path of weights")

    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg
    cfg.fold = parser_args.fold
    cfg.seed = parser_args.seed
    cfg.weights = parser_args.weights
    cfg.backbone = parser_args.backbone

    main(cfg)
