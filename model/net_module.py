import os
import torch.nn as nn
import numpy as np

from model.get_denoiser import get_denoiser

from model.net_module_part import H_layer
from model.net_module_part import Ht_layer
from model.net_module_part import prox_fid
from model.net_module_part import grad_fid
from model.net_module_part import grad_fid_and_norm


class net_module(nn.Module):
    def __init__(self, denoiser, kernel, crop_size, device):
        super().__init__()

        self.denoiser = denoiser

        self.sigma = 1
        self.tau = 1
        self.rho = 1

        self.grad = grad_fid_and_norm(kernel, crop_size, device)

    def forward(self, x, u, y):
        u_kari = u+self.sigma*x
        input_denoiser = (self.sigma+self.rho)**-1*u_kari
        output_denoiser = self.denoiser(input_denoiser)
        u_plus = u_kari-self.sigma * output_denoiser
        x = (x+self.tau*self.rho*x)-self.tau * \
            self.grad(x, y)-self.tau*(2*u_plus-u)
        return x, u_plus, input_denoiser, output_denoiser

    def install_para(self, kernel_name, noise_lev):
        self.sigma, self.rho = 0.5, 1
        self.grad.get_rho_th(1)

        # Load mu dictionary for parameter settings
        mu_dict_path = "mu_dict_MoLGrad.npy"
        if not os.path.exists(mu_dict_path):
            print(
                f"Warning: '{mu_dict_path}' not found. Cannot load mu parameters.")
            print("Using default mu value of 1.0")
            mu = 1.0
        else:
            try:
                mu_dict = np.load(mu_dict_path, allow_pickle=True).item()
                # Check if the kernel_name and noise level exist in the dictionary
                if kernel_name in mu_dict and f"noise{noise_lev}" in mu_dict[kernel_name]:
                    mu = mu_dict[kernel_name][f"noise{noise_lev}"]
                else:
                    print(
                        f"Warning: kernel_name '{kernel_name}' or noise_lev '{noise_lev}' not found in mu_dict.")
                    print("Using default mu value of 1.0")
                    mu = 1.0
            except Exception as e:
                print(f"Error loading mu_dict: {e}")
                print("Using default mu value of 1.0")
                mu = 1.0

        self.grad.get_mu(mu)
        self.tau = 0.8/(self.sigma+0.5*mu)
