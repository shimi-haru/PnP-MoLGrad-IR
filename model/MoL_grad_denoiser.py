import torch
import torch.nn as nn
import torch.nn.functional as F


class soft_shrink_fix(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.layer = nn.ReLU()
        self.alpha = alpha

    def forward(self, input):
        input_m = -input
        output = self.layer(input-self.alpha)
        output_m = -self.layer(input_m-self.alpha)
        return output+output_m


class sc_sigmoid_layer(nn.Module):

    def __init__(self, beta: int = 1) -> None:
        super().__init__()
        self.act = nn.Sigmoid()
        self.beta = beta

    def forward(self, input) -> torch.Tensor:
        return self.act(self.beta*input)


class reg_relu(nn.Module):
    def __init__(self, gamma=0.05, alpha=0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x):
        result = torch.where(x <= -self.gamma,
                             torch.zeros_like(x), x)
        mask = (x > -self.gamma) & (x < self.gamma)
        result = torch.where(
            mask,
            (x+self.gamma)**2/self.gamma/4,
            result
        )

        result = torch.where(x >= self.gamma,
                             x, result)

        return result-self.alpha


class reg_relu_custom(nn.Module):
    def __init__(self, gamma=0.1, alpha=0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.alpha = alpha

    def forward(self, x):
        gamma = F.softplus(100*self.gamma)/100 + 1e-6
        result = torch.where(x <= -gamma,
                             torch.zeros_like(x), x)
        mask = (x > -gamma) & (x < gamma)
        result = torch.where(
            mask,
            (x+gamma)**2/gamma/4,
            result
        )
        result = torch.where(x >= gamma,
                             x, result)

        return result-self.alpha+self.gamma/1e3


class hardsig_custom(nn.Module):
    def __init__(self, sup=4):
        super().__init__()
        self.relu = nn.ReLU()
        self.sup = sup

    def forward(self, x):
        return torch.where(x <= self.sup, self.relu(x), torch.ones_like(x)*self.sup)


class Res_block_en(nn.Module):
    def __init__(self, in_channels, beta, bias1=True, bias2=False,  act="soft", gamma=0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=bias1)
        if act == "soft":
            self.act = nn.Softplus(beta=beta)
        elif act == "reg_relu":
            self.act = reg_relu(gamma=gamma, alpha=0)
        elif act == "reg_relu_custom":
            self.act = reg_relu_custom(alpha=0)
        # self.act = custom_softplus(beta=beta)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=bias2)

    def forward(self, x):
        dif_input = self.conv1(x)
        skip_connect = self.act(dif_input)
        x1 = self.conv2(skip_connect)+x
        return x1


class block_en_2layer(nn.Module):
    def __init__(self, in_channels, beta, bias1=True, bias2=False, act="soft", gamma=0.05):
        super().__init__()
        self.res1 = Res_block_en(
            in_channels=in_channels, beta=beta, bias1=bias1, bias2=bias2, act=act, gamma=gamma)
        self.res2 = Res_block_en(
            in_channels=in_channels, beta=beta, bias1=bias1, bias2=bias2, act=act, gamma=gamma)

    def forward(self, x):
        x1 = self.res1(x)
        x1 = self.res2(x1)
        return x1


class pooling_en2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, input):
        return self.conv(self.pool(input))


class sc_hardsig(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.act = nn.Hardsigmoid()
        self.gamma = gamma

    def forward(self, x):
        return self.act(x*3/self.gamma)


class MoL_nonneg(nn.Module):
    def __init__(self, in_channels, beta=100, gamma=0.1, act="soft", bias=False, act_end="hardsig", bias_cons=False, mid_num=[64, 128, 256, 512]):
        super().__init__()

        self.beta = beta
        self.bias1 = True
        self.bias2 = True
        self.bias_end = bias

        self.val = False
        self.bias_cons = bias_cons

        # mid_num = [64, 128, 256, 512]

        self.inconv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=mid_num[0], kernel_size=3, padding=1, bias=False)

        self.res1_en = block_en_2layer(
            in_channels=mid_num[0], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)
        self.res11_en = block_en_2layer(
            in_channels=mid_num[0], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)
        self.downsamp2 = pooling_en2(
            in_channels=mid_num[0], out_channels=mid_num[1])
        self.res2_en = block_en_2layer(
            in_channels=mid_num[1], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)
        self.res22_en = block_en_2layer(
            in_channels=mid_num[1], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)
        self.downsamp3 = pooling_en2(
            in_channels=mid_num[1], out_channels=mid_num[2])
        self.res3_en = block_en_2layer(
            in_channels=mid_num[2], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)
        self.res33_en = block_en_2layer(
            in_channels=mid_num[2], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)
        self.downsamp4 = pooling_en2(
            in_channels=mid_num[2], out_channels=mid_num[3])
        self.res4_en = block_en_2layer(
            in_channels=mid_num[3], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)
        self.res44_en = block_en_2layer(
            in_channels=mid_num[3], beta=self.beta, bias1=self.bias1, bias2=self.bias2, act=act)

        if act_end == "hardsig":
            self.midact1 = sc_hardsig(gamma=gamma)
            self.midact2 = sc_hardsig(gamma=gamma)
            self.midact3 = sc_hardsig(gamma=gamma)
            self.midact4 = sc_hardsig(gamma=gamma)
        elif act_end == "relu":
            self.midact1 = hardsig_custom(sup=4)
            self.midact2 = hardsig_custom(sup=4)
            self.midact3 = hardsig_custom(sup=4)
            self.midact4 = hardsig_custom(sup=4)

        elif act_end == "sig":
            self.midact1 = sc_sigmoid_layer(beta=beta)
            self.midact2 = sc_sigmoid_layer(beta=beta)
            self.midact3 = sc_sigmoid_layer(beta=beta)
            self.midact4 = sc_sigmoid_layer(beta=beta)
        if self.bias_end:
            self.bias1 = bias_only_layer(mid_num[0])
            self.bias2 = bias_only_layer(mid_num[1])
            self.bias3 = bias_only_layer(mid_num[2])
            self.bias4 = bias_only_layer(mid_num[3])
            self.inconv2 = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=mid_num[0], kernel_size=3, padding=1, bias=True)
            self.midact = hardsig_custom(sup=4)

        # self.outconv = nn.Conv2d(in_channels=mid_num[0], out_channels=in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):

        # torch.set_grad_enabled(True)
        # x.requires_grad_(True)

        x11 = x
        x1 = self.inconv(x11)

        # x1_skip=self.inconv2(x1)

        x3_down1 = self.res1_en(x1)
        x1_down1 = self.res11_en(x3_down1)
        if self.bias_end:
            x1_down1 = self.bias1(x1_down1)
        x2_down1 = self.midact1(x1_down1)

        x2 = self.downsamp2(x1)
        x3_down2 = self.res2_en(x2)
        x1_down2 = self.res22_en(x3_down2)
        if self.bias_end:
            x1_down2 = self.bias2(x1_down2)
        x2_down2 = self.midact2(x1_down2)

        x3 = self.downsamp3(x2)
        x3_down3 = self.res3_en(x3)
        x1_down3 = self.res33_en(x3_down3)
        if self.bias_end:
            x1_down3 = self.bias3(x1_down3)
        x2_down3 = self.midact3(x1_down3)

        x4 = self.downsamp4(x3)
        x3_down4 = self.res4_en(x4)
        x1_down4 = self.res44_en(x3_down4)
        if self.bias_end:
            x1_down4 = self.bias4(x1_down4)
        x2_down4 = self.midact4(x1_down4)

        if self.bias_end:
            x22 = self.inconv2(x11)
            x33 = self.midact(x22)

            x1 = self.jacobian(x11, x1_down1, x2_down1)+self.jacobian(x11, x1_down2, x2_down2) + \
                self.jacobian(x11, x1_down3, x2_down3) + \
                self.jacobian(x11, x1_down4, x2_down4) + \
                self.jacobian(x11, x22, x33)
        else:
            x1 = self.jacobian(x11, x1_down1, x2_down1)+self.jacobian(x11, x1_down2, x2_down2) + \
                self.jacobian(x11, x1_down3, x2_down3) + \
                self.jacobian(x11, x1_down4, x2_down4)

        # torch.set_grad_enabled(False)

        x = x1
        return x

    def weight_const(self, layer):
        if isinstance(layer, nn.Conv2d):
            # if (layer is not self.outconv):
            layer.weight.data = torch.clamp(layer.weight.data, min=0)
            # a=None

    def bias_const(self, layer):
        if isinstance(layer, nn.Conv2d) and layer.bias is not None:
            layer.bias.data = torch.clamp(layer.bias.data, max=0)

    def apply_weight_constraints(self, do_nonneg):
        with torch.no_grad():
            if do_nonneg:
                self.apply(self.weight_const)
            if self.bias_cons:
                self.apply(self.bias_const)

    def jacobian(self, x, n, x1):
        return torch.autograd.grad(n, x, grad_outputs=x1, create_graph=True)[0]


class downsamp(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, input):
        return 4*self.pool(input)


class bias_only_layer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        return x+self.bias
