import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss
from math import exp
from torch.autograd import Variable


class SSIM_loss(_Loss):

    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, input, target):
        return 1-self.ssim(input, target)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(
        img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def PSNR(output_image, true_image, max_value=1.0):
    mse = F.mse_loss(output_image, true_image)
    if mse == 0:
        return float("inf")

    psnr = 20*torch.log10(max_value/torch.sqrt(mse))
    return psnr.item()


class get_jaco_norm(nn.Module):
    def __init__(self, size, device):
        super().__init__()

        self.size = size
        self.device = device
        vec = torch.ones(size).to(self.device)
        vec /= torch.norm(vec.view(size[0], -1),
                          dim=1, p=2).view(size[0], 1, 1, 1)
        self.vec = vec

    def forward(self, input, output, mode=[]):
        # outputt = input-output
        if mode == []:
            def operator(vec): return torch.autograd.grad(
                output, input, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
            spector_norm_jaco = self.power_iteration(
                operator, input.size(), num_ite=10)
        elif mode == "nonexp":
            outputt = input-output

            def operator(vec): return torch.autograd.grad(
                outputt, input, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
            spector_norm_jaco = self.power_iteration(
                operator, input.size(), num_ite=10)
        elif mode == "MMO":
            outputt = 2*output-input
            # outputt = output
            def operator(vec): return torch.autograd.grad(
                outputt, input, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
            spector_norm_jaco = self.power_iteration(
                operator, input.size(), num_ite=10)
        return spector_norm_jaco.max().item()

    def power_iteration(self, operator, size, num_ite):
        vec = self.vec

        with torch.no_grad():
            for i in range(num_ite):
                new_vec = operator(vec)
                new_vec = new_vec / torch.norm(new_vec.view(
                    size[0], -1), dim=1, p=2).view(size[0], 1, 1, 1)
                old_vec = vec
                vec = new_vec

        self.vec = vec
        new_vec = operator(vec)
        div = torch.norm(
            vec.view(size[0], -1), dim=1, p=2).view(size[0])
        eigenvalue = torch.abs(
            torch.sum(vec.view(size[0], -1) * new_vec.view(size[0], -1), dim=1)) / div
        # , torch.norm(vec.view(size[0], -1)-new_vec.view(size[0], -1), dim=1, p=2).view(size[0])
        return eigenvalue


class negative_weight_loss(nn.Module):
    def __init__(self, mse_w, sum_w):
        super().__init__()
        self.mseloss = nn.MSELoss()
        self.sum_w = sum_w
        self.sum_ww = sum_w
        self.mse_w = mse_w
        self.sum_th = 1000

    def sum_negative(self, model):
        negative_weight_sum = torch.tensor(
            [0.0], device=next(model.parameters()).device)
        positive_weight_sum = torch.tensor(
            [0.0], device=next(model.parameters()).device)

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # if isinstance(module, nn.ConvTranspose2d):

                if hasattr(model, 'outconv') and module is model.outconv:
                    # if (module is model.outconv):
                    # if module is model.inconv:
                    continue

                weights = module.weight
                negative_weights = weights[weights < 0]
                positive_weights = weights[weights > 0]

                negative_weight_sum += (negative_weights**2).sum()
                positive_weight_sum += (positive_weights**2).sum()

        return negative_weight_sum, positive_weight_sum

    def forward(self, output, target, model):
        sum_negative, sum_positive = self.sum_negative(model)
        sum_neg_clip = torch.min(sum_negative, torch.tensor(self.sum_th))
        mse = self.mseloss(output, target)
        return self.sum_w*sum_neg_clip+self.mse_w*mse, sum_negative, sum_positive, mse


class mse_loss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.sum_w = 0
        self.mse_w = 1
        self.sum_ww = 0

    def forward(self,  output, target, _):
        loss = self.loss(output, target)
        return loss, torch.tensor([0]), torch.tensor([0]), loss
