import torch
import torch.nn.functional as F
import numpy as np


from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.my_loss import PSNR
from utils.data_loading import data_loading_test
from utils.ssim import ssim
from utils.my_loss import get_jaco_norm
from utils.utils import imshow
from utils.utils import install_denoiser


def main_denoise():
    # Network model name to use for denoising
    net_name = "MoL_Grad_nonneg"

    # Score function type ("psnr", "ssim", or "mse")
    score_function = "psnr"

    # Flag to indicate whether to process grayscale images (True) or color images (False)
    gray = False

    # Crop size for input images (crop_size x crop_size)
    crop_size = 256

    # Noise level (standard deviation) of additive noise
    noise_lev = 0.15

    # Flag to process a single image (True) or compute average over all images (False)
    single_mode = False
    # Image index to process when single_mode is True
    image_num = 4

    # Flag to display results using imshow (True) or skip visualization (False)
    plot = False

    # Flag to compute Lipschitz constant (True) or skip (False)
    get_lip = True

    # List of folder paths containing test images
    test_path = ["../BSDS300/images/test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No blur operator for denoising task
    blur_mat = None
    # Random seed for reproducibility
    seed = 3

    dataset = data_loading_test(
        device, blur_mat, test_path, noise_lev, gray, crop_size, seed)
    dataloader = DataLoader(
        dataset, 1)

    denoiser = install_denoiser(net_name, gray, device)

    if get_lip:
        loss = get_jaco_norm(size=[1, 1 if gray else 3,
                                   crop_size, crop_size], device=device)
    denoiser.eval()

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
    lip_array = np.zeros(length_dataloader)
    lip = 0

    for i, batch in tqdm(enumerate(dataloader), total=length_dataloader, desc='Validation round', unit="batch", leave=False):

        if (not single_mode) or (single_mode and i == image_num):
            noise_image, true_image = batch["noise"], batch["true"]
            if noise_image.numel() == 0 or true_image.numel() == 0:
                print("skip")
            else:
                noise_image = noise_image.to(
                    device=device, dtype=torch.float32).requires_grad_(True)
                true_image = true_image.to(device=device, dtype=torch.float32)

                if get_lip:
                    # Enable gradient computation for Jacobian calculation
                    torch.set_grad_enabled(True)
                    denoise_image = denoiser(noise_image)

                    lip = loss(noise_image, denoise_image)
                    lip_array[i] = lip

                    # Explicitly discard gradients
                    if noise_image.grad is not None:
                        noise_image.grad.zero_()

                    denoise_image = denoise_image.detach()
                    torch.set_grad_enabled(False)
                else:
                    with torch.no_grad():
                        denoise_image = denoiser(noise_image)

                score_kari = score_f(denoise_image, true_image)
                score_noise_kari = score_f(noise_image.detach(), true_image)

                if single_mode and i == image_num:
                    break

                score += score_kari
                score_noise += score_noise_kari

    score = score/length_dataloader
    score_noise = score_noise/length_dataloader

    if single_mode:
        print(f"Noise image score ({score_function}): {score_noise_kari:.4f}")
        print(f"Restoration image score ({score_function}): {score_kari:.4f}")
    else:
        print(f"Noise image score ({score_function}): {score_noise:.4f}")
        print(f"Restoration image score ({score_function}): {score:.4f}")
    if get_lip:
        print(f"Lipschitz constant: {np.max(lip_array):.4f}")

    if plot:
        imshow(noise_image[0], title="True Image")
        imshow(true_image[0], title="Noisy Image")
        imshow(denoise_image[0], title="Restoration Image")


if __name__ == "__main__":
    main_denoise()
