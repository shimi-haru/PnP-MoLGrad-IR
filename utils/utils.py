import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from model.get_denoiser import get_denoiser


def install_denoiser(net_name, gray, device):

    denoiser = get_denoiser(net_name, image_channels=1 if gray else 3)

    file_path = "../audio/para_data/train_denoiser5/mol_grad_unet7/mol_grad_unet7_noise=0.15_notie.pth"
    if os.path.exists(file_path):
        state = torch.load(file_path, map_location=device)
        denoiser.load_state_dict(state)

    else:
        print(f"No checkpoint found at {file_path}")
        print(file_path)
    denoiser = denoiser.to(device)

    return denoiser


def imshow(tensor, title=None):
    # テンソルをCPUに移動し、NumPy配列に変換
    image = tensor.detach().cpu().numpy()

    # チャンネルの順序を (C, H, W) から (H, W, C) に変
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)

    # 画像の表示
    # image = Image.fromarray(image)
    plt.imshow(image, cmap='gray' if image.shape[2] == 1 else None)

    # メモリを削除
    plt.axis('off')

    # タイトルがあれば追加
    if title is not None:
        if title:
            plt.title(title, pad=20)  # padでタイトルの位置を調整
        else:
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
    else:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 画像表示
    plt.show()
