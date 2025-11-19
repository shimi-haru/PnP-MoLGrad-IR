import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp


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


class SSIM(torch.nn.Module):
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


"""
path1 = "..//BSDS300/images/test/3096.jpg"
path2 = "..//BSDS300/images/test/8023.jpg"
img1 = Image.open(path1)
img2 = Image.open(path2)
image_size = 256
transform = transforms.Compose([
    transforms.CenterCrop(size=image_size),
    transforms.ToTensor()
])

ite_num = 10
noise_ar = np.zeros(ite_num)
ssim1 = np.zeros(ite_num)
ssim2 = np.zeros(ite_num)

for i in range(ite_num):
    noise_lev = 0.03*1.3**i
    noise_ar[i] = noise_lev

    add_noise = AddGaussianNoise(0, noise_lev)
    image1 = transform(img1).unsqueeze(0)
    image2 = transform(img2).unsqueeze(0)

    image = torch.cat([image1, image2])
    image_noise = add_noise(image)
    print(image.size())

    # plt.imshow(image_noise[0].permute(1, 2, 0))
    # plt.axis('off')
    # plt.show()

    # ssim_loss = SSIM()
    # ssim_out = ssim_loss(image[0].unsqueeze(0), image_noise[0].unsqueeze(0))
    ssim_out = ssim(image[0].unsqueeze(0), image_noise[0].unsqueeze(0))
    print(ssim_out.item())
    ssim1[i] = ssim_out.item()

    ssim_val, ssim_image = ssim_sk(
        image[0].permute(1, 2, 0).numpy(), image_noise[0].permute(1, 2, 0).numpy(), channel_axis=-1, full=True, data_range=1)
    print(ssim_val.item())
    ssim2[i] = ssim_val.item()

plt.plot(noise_ar, ssim1, label="code")
plt.plot(noise_ar, ssim2, label="library")
plt.legend()
plt.show()
"""
