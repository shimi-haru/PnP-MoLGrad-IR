import torch
import torch.nn as nn


def get_eig_H(kernel, kernel_size, image_size):
    kernel_row = kernel.flatten()
    root_row = torch.zeros(kernel_row.shape, dtype=torch.complex64)
    m = kernel_size
    n = image_size
    for i in range(m):
        root_row[i*m:(i+1)*m] = torch.exp(2*torch.pi*1j/n**2 *
                                          (torch.arange(i*n, n*i+m)-((n*(kernel_size//2))+kernel_size//2)))
    eig_row = torch.zeros(n**2, dtype=torch.complex64)
    half_n = n**2//2
    for i in range(half_n):
        eig_row[i] = torch.sum(kernel_row*(root_row**(i)))
    eig_row[0] = torch.real(torch.sum(kernel_row))
    eig_row[half_n] = torch.real(torch.sum(kernel_row*(root_row**(half_n))))
    eig_row[half_n+1:] = torch.conj(torch.flip(eig_row[1:half_n], dims=[0]))
    return eig_row


def conv_tensor(image_tensor, eig_row):
    """
    Apply convolution in frequency domain using eigenvalues.
    Equivalent to circular convolution with kernel having eigenvalues eig_row.

    Args:
        image_tensor: Input image (batch_size, color_size, height, width)
        eig_row: Eigenvalues of convolution matrix (complex64, n²)

    Returns:
        result_tensor: Convolved image (batch_size, color_size, height, width)
    """
    # Get image dimensions
    batch_size, color_size, image_h, image_w = image_tensor.shape
    assert image_h == image_w, "Only square images supported"

    N = image_h

    # Convert to complex dtype for FFT computation
    image_tensor = image_tensor.to(dtype=torch.complex64)

    # Flatten spatial dimensions and apply FFT
    tensor_flat = image_tensor.view(batch_size, color_size, -1)
    fft_tensor = torch.fft.fft(tensor_flat, dim=-1)

    # Reshape eig_row for broadcasting (1, 1, n²)
    eig_row = eig_row.to(image_tensor.device).to(torch.complex64)
    eig_row = eig_row.view(1, 1, -1)

    # Frequency-domain multiplication (element-wise product with eigenvalues)
    complex_tensor = fft_tensor * eig_row

    # Convert back to spatial domain via inverse FFT
    ifft_tensor = torch.fft.ifft(complex_tensor, dim=-1)

    # Extract real part and reshape back to original dimensions
    result_tensor = torch.real(ifft_tensor).view(batch_size, color_size, N, N)

    return result_tensor


class H(nn.Module):
    """
    Convolution operator H defined by kernel in frequency domain.
    """

    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.kernel = kernel
        if kernel.numel() > 0:
            # Precompute eigenvalues of convolution matrix
            self.eig_row = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            self.image_size = image_size
            self.convlayer = Conv2d(self.eig_row)

    def forward(self, image):
        """Apply convolution H to image."""
        if self.kernel.numel() > 0:
            return self.convlayer(image)
        else:
            return image


class H_layer(nn.Module):
    """
    Wrapper for H operator that ensures real-valued output.
    """

    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.H = H(kernel, image_size, device)

    def forward(self, image):
        """Apply H and extract real part."""
        return torch.real(self.H(image))


class Ht(nn.Module):
    """
    Adjoint (transpose) convolution operator H^T.
    Uses conjugate of eigenvalues.
    """

    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.kernel = kernel
        if kernel.numel() > 0:
            self.eig_row = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            self.image_size = image_size
            # Use conjugate eigenvalues for adjoint
            self.convtlayer = Convtrans2d(self.eig_row)

    def forward(self, image):
        """Apply adjoint convolution H^T to image."""
        if self.kernel.numel() > 0:
            return self.convtlayer(image)
        else:
            return image


class Ht_layer(nn.Module):
    """
    Wrapper for Ht operator that ensures real-valued output.
    """

    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.Ht = Ht(kernel, image_size, device)

    def forward(self, image):
        """Apply H^T and extract real part."""
        return torch.real(self.Ht(image))


class Conv2d(nn.Module):
    """
    Frequency-domain convolution using precomputed eigenvalues.
    """

    def __init__(self, eig_row):
        super().__init__()
        self.eig_row = eig_row

    def forward(self, image):
        """Apply frequency-domain convolution."""
        return conv_tensor(image, self.eig_row)


class Convtrans2d(nn.Module):
    """
    Adjoint (transpose) frequency-domain convolution.
    Uses conjugate of eigenvalues.
    """

    def __init__(self, eig_row):
        super().__init__()
        # Store conjugate for adjoint operation
        self.eig_row = torch.conj(eig_row)

    def forward(self, image):
        """Apply adjoint convolution using conjugate eigenvalues."""
        return conv_tensor(image, self.eig_row)


class prox_fid(nn.Module):
    """
    Proximal operator for fidelity term: 0.5 * ||Hx - y||^2
    prox_γf(x) = (I + γ H^T H)^{-1} (γ H^T y + x)

    Computed efficiently in frequency domain using eigenvalues.
    """

    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.kernel = kernel
        self.device = device
        self.gamma = 0.8  # step size parameter

        if kernel.numel() > 0:
            # Precompute eigenvalues of H
            self.eig_row_H = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            # Eigenvalues of (I + γ H^T H)^{-1} in frequency domain
            self.eig_row_inv = 1 / \
                (1 + self.gamma * torch.abs(self.eig_row_H)**2).to(device)
            self.inv_layer = Conv2d(self.eig_row_inv)
            self.trans_layer = Convtrans2d(self.eig_row_H)

    def forward(self, x, y):
        """
        Compute prox_γf(x) = (I + γ H^T H)^{-1} (γ H^T y + x)
        """
        if self.kernel.numel() > 0:
            return torch.real(self.inv_layer(self.gamma * self.trans_layer(y) + (x + 0j)))
        else:
            # No convolution: prox becomes (1 + γ)^{-1} (γ y + x)
            return (1 + self.gamma) * (self.gamma * y + x)

    def get_gamma(self, gamma):
        """Update step size parameter and recompute inverse eigenvalues."""
        self.gamma = gamma
        if self.kernel.numel() > 0:
            self.eig_row_inv = 1 / \
                (1 + self.gamma * torch.abs(self.eig_row_H)**2).to(self.device)
            self.inv_layer = Conv2d(self.eig_row_inv)


class grad_fid(nn.Module):
    """
    Gradient of fidelity term: ∇f(x) = μ H^T (Hx - y)

    Args:
        mu: Scaling parameter for gradient
    """

    def __init__(self, kernel, crop_size, device):
        super().__init__()
        self.kernel = kernel
        self.H = H(self.kernel, crop_size, device)
        self.TransH = Ht(self.kernel, crop_size, device)
        self.mu = 1.0  # Initial scaling parameter

        if kernel.numel() > 0:
            self.kernel_size = kernel.shape[-1]
        else:
            self.kernel_size = 0

    def forward(self, x, y):
        """Compute gradient μ H^T (Hx - y)."""
        return torch.real(self.mu * self.TransH(self.H(x) - (y + 0j)))

    def get_mu(self, mu):
        """Update scaling parameter mu."""
        self.mu = mu


class Pm_set(nn.Module):
    """
    Container for projection operators Pm (onto null space) and Pm_perp (onto range).
    """

    def __init__(self, kernel, image_size, device, th):
        super().__init__()
        # Projection onto orthogonal complement of null space
        self.Pm_perp = Pm_perp(kernel, image_size, device, th)
        # Projection onto null space
        self.Pm = Pm(kernel, image_size, device, th)

    def forward(self):
        pass

    def get_rho_th(self, th):
        """Update threshold for null space projections."""
        self.Pm_perp.get_th(th)
        self.Pm.get_th(th)


class grad_fid_and_norm(nn.Module):
    """
    Combined gradient with regularization term.
    Supports multiple variants (alpha=0..12) for different regularization strategies.

    Typical form: μ H^T(Hx - y) + ρ_th * R(x)
    where R(x) depends on alpha parameter.
    """

    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.kernel = kernel
        self.image_size = image_size
        self.mu = 1.0  # Fidelity weight
        self.rho_th = 0.2  # Regularization weight

        if kernel.numel() > 0:
            self.kernel_size = kernel.shape[-1]
            # Precompute eigenvalues of H
            self.eig_row = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)

            # Soft-thresholding filter for frequency components below threshold
            self.eig_th_pre = (torch.where(
                torch.abs(self.eig_row)**2 > (self.rho_th / self.mu),
                torch.tensor(0), torch.tensor(1)) + 0j)
            self.eig_th = self.eig_th_pre * \
                (self.rho_th / self.mu - torch.abs(self.eig_row)**2) / \
                (self.rho_th / self.mu)
            self.eig_th2 = torch.conj(self.eig_th_pre *
                                      (self.rho_th / self.mu - torch.abs(self.eig_row)) /
                                      (self.rho_th / self.mu) * torch.exp(1j * torch.angle(self.eig_row)))

            # Phase-only filter
            self.eig_th3 = self.eig_th_pre * \
                torch.exp(1j * torch.angle(self.eig_row))
            self.eig_th3_conj = self.eig_th_pre * \
                torch.exp(-1j * torch.angle(self.eig_row))

            # Precompute convolution layers
            self.convlayer4 = Conv2d(self.eig_th3)
            self.convlayer4_conj = Conv2d(self.eig_th3_conj)
            self.H = Conv2d(self.eig_row)
            self.transH = Conv2d(torch.conj(self.eig_row))
            self.convlayer2 = Conv2d(self.eig_th)
            self.convlayer3 = Conv2d(self.eig_th2)

            # Projection operators
            self.Pm_set = Pm_set(kernel, image_size,
                                 device, self.rho_th / self.mu)
        else:
            self.kernel_size = 0

        self.alpha = []  # Variant selector

    def forward(self, x, y):
        """
        Compute combined gradient with different regularization variants.

        Args:
            x: Current image estimate
            y: Observation (noisy/blurry image)

        Returns:
            Gradient direction combining fidelity and regularization
        """
        if self.kernel.numel() > 0:
            # Variant-specific computation
            if self.alpha == []:
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.Pm_set.Pm_perp(x)))
            elif self.alpha == 1:
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.convlayer2(x) - self.convlayer3(y)))
            elif self.alpha == 2:
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.convlayer2(x) - self.convlayer2(y)))
            elif self.alpha == 3:
                return torch.real(self.mu * self.Pm_set.Pm(self.transH(self.H(self.Pm_set.Pm(x)) - y)) +
                                  self.rho_th * (self.Pm_set.Pm_perp(x - y)))
            elif self.alpha == 4:
                return torch.real(self.mu * self.Pm_set.Pm(self.transH(self.H(self.Pm_set.Pm(x)) - y)) +
                                  self.rho_th * (self.Pm_set.Pm_perp(x)))
            elif self.alpha == 5:
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.convlayer2(x)))
            elif self.alpha == 6:
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.Pm_set.Pm_perp(x - y)))
            elif self.alpha == 7:
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.Pm_set.Pm_perp(x)))
            elif self.alpha == 9:
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.convlayer2(x) - self.Pm_set.Pm_perp(self.transH(y))))
            elif self.alpha == 10:
                return torch.real(self.mu * self.Pm_set.Pm(self.transH(self.H(self.Pm_set.Pm(x)) - y)) +
                                  self.rho_th * (self.Pm_set.Pm_perp(x)) - self.mu * self.Pm_set.Pm_perp(self.transH(y)))
            elif self.alpha == 11:
                return torch.real(self.mu * (self.Pm_set.Pm(self.transH(self.H(self.Pm_set.Pm(x)) - y))) +
                                  self.rho_th * (self.convlayer4_conj(self.convlayer4(x))) -
                                  self.rho_th**0.5 * self.convlayer4_conj(y))
            elif self.alpha == 12:
                return torch.real(self.mu * ((self.transH(self.H(x) - y))) +
                                  self.rho_th * (self.convlayer4_conj(self.convlayer4(x))) -
                                  self.rho_th**0.5 * self.convlayer4_conj(y))
            else:
                # Default variant
                return torch.real(self.mu * self.transH(self.H(x) - y) +
                                  self.rho_th * (self.Pm_set.Pm_perp(x)))
        else:
            # No convolution: simple gradient
            return self.mu * (x - y)

    def get_rho_th(self, rho_th):
        """Update regularization weight and recompute frequency filters."""
        if self.kernel.numel() > 0:
            self.rho_th = rho_th
            # Recompute soft-thresholding filters
            self.eig_th_pre = (torch.where(
                torch.abs(self.eig_row)**2 > (self.rho_th / self.mu),
                torch.tensor(0), torch.tensor(1)) + 0j)
            self.eig_th = self.eig_th_pre * \
                (self.rho_th / self.mu - torch.abs(self.eig_row)**2) / \
                (self.rho_th / self.mu)
            self.convlayer2 = Conv2d(self.eig_th)
            self.eig_th2 = torch.conj(self.eig_th_pre *
                                      (self.rho_th / self.mu - torch.abs(self.eig_row)) /
                                      (self.rho_th / self.mu) * torch.exp(1j * torch.angle(self.eig_row)))
            self.convlayer3 = Conv2d(self.eig_th2)

            # Update projection operators
            self.Pm_set.get_rho_th(self.rho_th / self.mu)

            # Update phase-only filters
            self.eig_th3 = self.eig_th_pre * \
                torch.exp(1j * torch.angle(self.eig_row))
            self.eig_th3_conj = self.eig_th_pre * \
                torch.exp(-1j * torch.angle(self.eig_row))
            self.convlayer4 = Conv2d(self.eig_th3)
            self.convlayer4_conj = Conv2d(self.eig_th3_conj)

    def get_mu(self, mu):
        """Update fidelity weight and recompute frequency filters."""
        if self.kernel.numel() > 0:
            self.mu = mu
            if mu == 0:
                self.mu = 1e-5  # Prevent division by zero

            # Recompute all frequency-dependent filters
            self.eig_th_pre = (torch.where(
                torch.abs(self.eig_row)**2 > (self.rho_th / self.mu),
                torch.tensor(0), torch.tensor(1)) + 0j)
            self.eig_th = self.eig_th_pre * \
                (self.rho_th / self.mu - torch.abs(self.eig_row)**2) / \
                (self.rho_th / self.mu)
            self.convlayer2 = Conv2d(self.eig_th)
            self.eig_th2 = torch.conj(self.eig_th_pre *
                                      (self.rho_th / self.mu - torch.abs(self.eig_row)) /
                                      (self.rho_th / self.mu) * torch.exp(1j * torch.angle(self.eig_row)))
            self.convlayer3 = Conv2d(self.eig_th2)

            # Update projection operators
            self.Pm_set.get_rho_th(self.rho_th / self.mu)

            # Update phase-only filters
            self.eig_th3 = self.eig_th_pre * \
                torch.exp(1j * torch.angle(self.eig_row))
            self.eig_th3_conj = self.eig_th_pre * \
                torch.exp(-1j * torch.angle(self.eig_row))
            self.convlayer4 = Conv2d(self.eig_th3)
            self.convlayer4_conj = Conv2d(self.eig_th3_conj)


class Pm_perp(nn.Module):
    """
    Projection onto orthogonal complement of null space.
    Keeps frequency components with |H(ω)| < threshold.
    """

    def __init__(self, kernel, image_size, device, th):
        super().__init__()
        self.kernel = kernel
        self.image_size = image_size
        self.th = th
        self.device = device

        if kernel.numel() > 0:
            self.kernel_size = kernel.shape[-1]
            self.eig_row = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            # Binary mask: 1 where |H| < threshold, 0 otherwise
            self.eig_th = torch.where(torch.abs(self.eig_row)**2 < self.th,
                                      torch.tensor(1) + 0j, torch.tensor(0))
            self.Pm = Conv2d(self.eig_th)
        else:
            self.kernel_size = 0

    def forward(self, x):
        """Apply orthogonal complement projection."""
        return torch.real(self.Pm(x))

    def get_th(self, th):
        """Update threshold and recompute projection."""
        self.th = th
        if self.kernel.numel() > 0:
            self.kernel_size = self.kernel.shape[-1]
            self.eig_row = get_eig_H(
                self.kernel, self.kernel.shape[-1], self.image_size).to(self.device)
            # Recompute binary mask with new threshold
            self.eig_th = torch.where(torch.abs(self.eig_row)**2 < self.th,
                                      torch.tensor(1) + 0j, torch.tensor(0))
        self.Pm = Conv2d(self.eig_th)


class Pm(nn.Module):
    """
    Projection onto null space.
    Keeps frequency components with |H(ω)| > threshold.
    """

    def __init__(self, kernel, image_size, device, th):
        super().__init__()
        self.kernel = kernel
        self.image_size = image_size
        self.th = th
        self.device = device

        if kernel.numel() > 0:
            self.kernel_size = kernel.shape[-1]
            self.eig_row = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            # Binary mask: 1 where |H| > threshold, 0 otherwise
            self.eig_th = torch.where(torch.abs(self.eig_row)**2 > self.th,
                                      torch.tensor(1) + 0j, torch.tensor(0))
            self.Pm = Conv2d(self.eig_th)
        else:
            self.kernel_size = 0

    def forward(self, x):
        """Apply null space projection."""
        return torch.real(self.Pm(x))

    def get_th(self, th):
        """Update threshold and recompute projection."""
        self.th = th
        if self.kernel.numel() > 0:
            self.kernel_size = self.kernel.shape[-1]
            self.eig_row = get_eig_H(
                self.kernel, self.kernel.shape[-1], self.image_size).to(self.device)
            # Recompute binary mask with new threshold
            self.eig_th = torch.where(torch.abs(self.eig_row)**2 > self.th,
                                      torch.tensor(1) + 0j, torch.tensor(0))
        self.Pm = Conv2d(self.eig_th)
