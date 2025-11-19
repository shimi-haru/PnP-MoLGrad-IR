import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader


from model.net_module import net_module
from utils.my_loss import PSNR
from utils.data_loading import data_loading_test
from model.net_module_part import H_layer
from make_kernel import get_kernel
from utils.utils import imshow
from utils.utils import install_denoiser
from utils.ssim import ssim


def main_deblur():
    net_name = "MoL_Grad_nonneg"
    score_function = "psnr"

    # Flag to indicate whether to process grayscale images (True) or color images (False)
    gray = False

    # Crop size for input images (crop_size x crop_size)
    crop_size = 256

    # Noise level (standard deviation) added to the blurred images
    noise_lev = 0.03

    kernel_name = "real_kernel_small_1"
    kernel = get_kernel(kernel_name)

    # Number of iterations for the deblurring algorithm
    ite_num = 500

    # Flag to process a single image (True) or compute average over all images (False)
    single_mode = True
    # Image index to process when single_mode is True
    image_num = 0

    plot = False

    # List of folder paths containing test images
    test_path = ["../BSDS300/images/test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blur_mat = H_layer(kernel, crop_size, device).to(device)
    seed = None

    dataset = data_loading_test(
        device, blur_mat, test_path, noise_lev, gray, crop_size, seed)
    dataloader = DataLoader(
        dataset, 1)

    denoiser = install_denoiser(net_name, gray, device)
    model = net_module(
        denoiser, kernel, crop_size, device)

    model.eval()

    def score_f(x, y):
        if score_function == "psnr":
            return PSNR(x, y)
        if score_function == "ssim":
            return ssim(x, y).item()
        elif score_function == "mse":
            return F.mse_loss(x, y).item()

    score = 0
    score_noise = 0

    length_dataloader = len(dataloader)

    for i, batch in tqdm(enumerate(dataloader), total=length_dataloader, desc='Validation round', unit="batch", leave=False):

        model.install_para(kernel_name, noise_lev)

        if (not single_mode) or (single_mode and i == image_num):
            blur_image, true_image = batch["noise"], batch["true"]
            if blur_image.numel() == 0 or true_image.numel() == 0:
                print("skip")
            else:
                blur_image = blur_image.to(
                    device=device, dtype=torch.float32)
                true_image = true_image.to(device=device, dtype=torch.float32)

                x = torch.zeros(blur_image.size()).to(
                    device=device, dtype=torch.float32)
                u = torch.zeros(blur_image.size()).to(
                    device=device, dtype=torch.float32)

                for _ in range(ite_num):
                    torch.set_grad_enabled(True)
                    x.requires_grad_(True)
                    u.requires_grad_(True)
                    x_kari, u_kari, _, _ = model(x, u, blur_image)

                    x = x_kari.detach()
                    u = u_kari.detach()
                    torch.set_grad_enabled(False)

                restoration_image = x
                score_kari = score_f(restoration_image, true_image)
                score_noise_kari = score_f(blur_image, true_image)

                if single_mode and i == image_num:
                    break

                score += score_kari
                score_noise += score_noise_kari

            if plot:
                imshow(blur_image[0], title="True Image")
                imshow(true_image[0], title="Blurred Image")
                imshow(restoration_image[0], title="Restoration Image")

    score = score/length_dataloader
    score_noise = score_noise/length_dataloader

    if single_mode:
        print(
            f"Blurred image score ({score_function}): {score_noise_kari:.4f}")
        print(f"Restoration image score ({score_function}): {score_kari:.4f}")
    else:
        print(f"Blurred image score ({score_function}): {score_noise:.4f}")
        print(f"Restoration image score ({score_function}): {score:.4f}")


if __name__ == "__main__":
    main_deblur()
