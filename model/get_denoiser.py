from model.MoL_grad_denoiser import soft_shrink_fix

from model.MoL_grad_denoiser import MoL_nonneg


def get_denoiser(net_name, image_channels):

    if net_name in ("MoL_Grad_nonneg"):
        denoiser = MoL_nonneg(
            image_channels, gamma=0.05, act="reg_relu", bias=True, act_end="relu")

    elif net_name in ("soft_shrinkage", "TV"):
        denoiser = soft_shrink_fix(alpha=0.001)

    return denoiser
