import numpy as np
import torch
import torch.nn as nn
import math
import time
import logging
# import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
import os
import sys

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from pathlib import Path

from utils.data_loading import data_loading
from utils.data_loading import data_loading_test
from utils.my_loss import PSNR
from utils.my_loss import negative_weight_loss
from model.get_denoiser import get_denoiser


def custom_collate_fn(batch):
    batch = [item for item in batch if item and isinstance(
        item, dict) and 'true' in item]

    batch_size = 20

    if len(batch) < batch_size:
        missing_count = batch_size - len(batch)
        batch.extend([batch[-1]] * missing_count)

    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    # パラメータ設定
    net_name = "mol_grad_unet7"
    crop_size = 128
    noise_lev = 0.05
    gray = False
    use_gpu = True
    use_wandb = True

    device = torch.device("cuda" if use_gpu else "cpu")

    # デノイザーの取得
    denoiser = get_denoiser(net_name, image_channels=1 if gray else 3)
    denoiser.to(device=device)

    # パス設定
    train_path = ["../Flickr2K", "../DIV2K",
                  "../Waterloo Exploration Database"]
    test_path = ["../BSDS300/images/train"]

    # 訓練パラメータ
    epoch_size = 500
    batch_size = 20
    learning_rate = 1e-4
    weight_decay = 5e-7
    patience = 3
    grad_clip_th = 1e-1

    save_checkpoint = False
    content = "train"
    checkpoint_state = None

    wandb_project_name = "train_denoiser"
    wandb_run_name = net_name + "_noise=" + str(noise_lev)

    checkpoint_path = "./para_data/" + wandb_project_name + "/" + net_name
    checkpoint_path_debug = "./para_debug/" + wandb_project_name + "/" + net_name

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(checkpoint_path_debug, exist_ok=True)

    # データローダの作成
    train_dataset = data_loading(
        device, None, train_path, noise_lev, gray, crop_size)

    test_dataset = data_loading_test(device, None,
                                     test_path, noise_lev, gray, crop_size)

    train_dataloader = DataLoader(
        train_dataset, batch_size, drop_last=True, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(
        test_dataset, 1, drop_last=True)

    # wandbの初期化
    if use_wandb:
        experiment = wandb.init(
            project=wandb_project_name, name=content+"_"+wandb_run_name, settings=wandb.Settings(save_code=False))
        experiment.config.update(dict(epochs=epoch_size,
                                      noise_lev=noise_lev,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      save_checkpoint=True)
                                 )
    else:
        experiment = None

    # オプティマイザと損失関数の設定
    optimizer = optim.Adam(
        denoiser.parameters(), lr=learning_rate, weight_decay=weight_decay)

    grad_scaler = torch.cuda.amp.GradScaler(
        enabled=True if device.type != "cpu" else False)

    mse_w = 1
    sum_w = 0.01
    criterion = negative_weight_loss(mse_w, sum_w)

    # 訓練用の変数
    global_step = 0
    step = 0
    val_check_interval = 1
    lr_dec_epochnum = epoch_size * 8 / 10
    dec_lr = 5
    do_nonneg = False
    flag = False
    epoch_th = 50
    end_weight = epoch_size / 10
    sum_neg_th = 5e-4

    # 訓練ループ
    for epoch in range(1, epoch_size + 1):
        start_time = time.time()
        denoiser.train()

        batch_step = 0

        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epoch_size}', unit='img', leave=False, disable=True) as pbar:

            for batch in train_dataloader:
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
                    noise_image, true_image = batch["noise"], batch["true"]
                    noise_image = noise_image.to(
                        device=device, dtype=torch.float32).requires_grad_(True)
                    true_image = true_image.to(
                        device=device, dtype=torch.float32).requires_grad_(True)

                    # 順伝播
                    restoration_image = denoiser(noise_image)

                    # 損失計算
                    loss, sum_neg, sum_positive, mse = criterion(
                        restoration_image, true_image, denoiser)

                    # 逆伝播と最適化
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        denoiser.parameters(), grad_clip_th)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(noise_image.shape[0])
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    batch_step += 1
                    global_step += 1
                    step += 1

                    # 重み制約の適用
                    if not flag and (epoch >= end_weight or (sum_neg < sum_neg_th) and (epoch > 1)):
                        do_nonneg = True
                        criterion.sum_w = 0
                        criterion.mse_w = 1
                        flag = True

                    denoiser.apply_weight_constraints(do_nonneg)

                    # wandbへのログ
                    if use_wandb:
                        experiment.log({
                            "train loss": round(loss.item(), 6),
                            "epoch": epoch,
                            "train_noiselev": (torch.sum(batch["noise_lev"])/batch_size).item(),
                            "sum_nonneg": round(sum_neg.item(), 6),
                            "sum_positive": round(sum_positive.item()),
                            "mseloss": round(mse.item(), 6) if mse < 0.1 else 0.1,
                            "neg_weight": criterion.sum_w
                        })

                    # 評価
                    if (batch_step % math.floor(math.floor(len(train_dataset)/batch_size)*val_check_interval)) == 0:

                        denoiser.eval()
                        val_score = 0
                        val_psnr = 0
                        val_noise = 0

                        with torch.no_grad():
                            for i, eval_batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Validation round', unit="batch", leave=False, disable=True):

                                eval_noise_image, eval_true_image = eval_batch["noise"], eval_batch["true"]
                                eval_noise_image = eval_noise_image.to(
                                    device=device, dtype=torch.float32).requires_grad_(True)
                                eval_true_image = eval_true_image.to(
                                    device=device, dtype=torch.float32)

                                denoiser.val = True
                                eval_denoise_image = denoiser(eval_noise_image)
                                denoiser.val = False

                                eval_score_kari = F.mse_loss(
                                    eval_true_image, eval_denoise_image)
                                eval_score_noise_kari = F.mse_loss(
                                    eval_true_image, eval_noise_image)
                                eval_score_kari2 = PSNR(
                                    eval_true_image, eval_denoise_image)

                                val_score += eval_score_kari
                                val_noise += eval_score_noise_kari
                                val_psnr += eval_score_kari2

                        denoiser.train()
                        val_psnr = val_psnr / len(test_dataset)

                        if use_wandb:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'val_score': round(val_score.item(), 4) if val_score.item() < 3 else 3,
                                'noise_score': val_noise,
                                "val_PSNR": val_psnr
                            })

        # 学習率の減衰
        if (epoch % lr_dec_epochnum) == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / dec_lr

        # チェックポイントの保存
        if (epoch % epoch_th) == 0 and not epoch == 0:
            state_dict = denoiser.state_dict()
            torch.save(state_dict, str(checkpoint_path_debug +
                                       "/"+content+"_"+wandb_run_name+"_epoch_"+str(epoch)+".pth"))

    # 最終チェックポイントの保存
    if epoch <= epoch_th:
        state_dict = denoiser.state_dict()
        torch.save(state_dict, str(checkpoint_path_debug +
                                   "/"+content+"_"+wandb_run_name+"_end_epoch_"+str(epoch)+".pth"))

    if save_checkpoint:
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        state_dict = denoiser.state_dict()
        torch.save(state_dict, str(checkpoint_path +
                                   "/"+wandb_run_name+".pth"))

    if use_wandb and experiment is not None:
        experiment.finish()
