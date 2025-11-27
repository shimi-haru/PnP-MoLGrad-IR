import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import urllib.request

from model.get_denoiser import get_denoiser


def load_pth_from_github(url, save_path, map_location="cpu"):
    if not os.path.exists(save_path):
        print(f"Downloading: {url}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(url, save_path)

    print(f"Loading model from {save_path}")
    return torch.load(save_path, map_location=map_location)


def install_denoiser(net_name, gray, device):

    denoiser = get_denoiser(net_name, image_channels=1 if gray else 3)

    url = "https://github.com/shimi-haru/PnP-MoLGrad-IR/releases/download/v1.0.0/MoL_Grad_noise015.pth"
    save_path = "./param/MoL_Grad_noise015.pth"

    state = load_pth_from_github(url, save_path, map_location=device)
    denoiser.load_state_dict(state)
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
