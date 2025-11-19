import torch


def make_gaussblurk(n, sigma):

    # sigma = 4

    ax = torch.linspace(-(n//2), n//2, n)
    x, y = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(x**2+y**2)/(2*sigma**2))
    kernel = kernel/torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)


def make_identityk(n):
    kernel = torch.zeros([n, n])
    kernel[n//2, n//2] = 1
    return kernel.unsqueeze(0).unsqueeze(0)


def make_squarek(n):
    kernel = torch.ones([n, n])
    kernel = kernel/torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)


def make_gauss2(kernel_size, sigma, sigma_rate, theta):

    theta = torch.tensor([theta*torch.pi])
    sigma_x = sigma
    sigma_y = sigma*sigma_rate
    x = torch.arange(-(kernel_size // 2), kernel_size //
                     2 + 1, dtype=torch.float32)
    y = torch.arange(-(kernel_size // 2), kernel_size //
                     2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y, indexing='ij')

    # Rotate coordinates
    x_rot = x * torch.cos(theta) + y * torch.sin(theta)
    y_rot = -x * torch.sin(theta) + y * torch.cos(theta)

    # Calculate the Gaussian function with the rotated coordinates
    gauss = torch.exp(-((x_rot**2) / (2 * sigma_x**2) +
                        (y_rot**2) / (2 * sigma_y**2)))

    # Normalize the kernel
    gauss /= gauss.sum()

    return gauss.unsqueeze(0).unsqueeze(0)


def get_kernel(kernel_name):
    kernel_size = 13
    if kernel_name == "gauss_blur":
        sigma = 0.5
        kernel_size = 13
        kernel = make_gaussblurk(kernel_size, sigma)
    elif kernel_name == "gauss_blur1":
        sigma = 1.6
        kernel_size = 25
        kernel = make_gaussblurk(kernel_size, sigma)
    elif kernel_name == "gauss_blur2":
        sigma = 0.7
        kernel_size = 25
        kernel = make_gaussblurk(kernel_size, sigma)
    elif kernel_name == "gauss_blur3":
        sigma = 1
        kernel_size = 25
        kernel = make_gaussblurk(kernel_size, sigma)
    elif kernel_name == "square":
        kernel_size = 5
        kernel = make_squarek(kernel_size)
    elif kernel_name == "noise_only":
        # kernel = make_identityk(1)
        kernel = torch.tensor([])
    elif kernel_name == "noise_only_pnp":
        # kernel = make_identityk(1)
        kernel = torch.tensor([])
    elif "real_" in kernel_name:
        path = "blur_kernel/"+kernel_name+".pt"
        kernel = torch.load(path).unsqueeze(0).unsqueeze(0)
    elif kernel_name == "gauss2_blur1":
        sigma = 1
        kernel_size = 25
        theta = 0.25
        sigma_rate = 0.7
        kernel = make_gauss2(kernel_size, sigma, sigma_rate, theta)
    elif kernel_name == "gauss2_blur2":
        kernel_size = 25
        sigma = 1.6
        sigma_rate = 0.5
        theta = -0.25
        kernel = make_gauss2(kernel_size, sigma, sigma_rate, theta)
    return kernel
