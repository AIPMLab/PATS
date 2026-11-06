"""Learnable edge-aware Bezier transform (E-SBCT) used by PATS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _gaussian_kernel_2d(size=7, sigma=1.5):
    """Return a 2D Gaussian kernel as [1, 1, size, size]."""
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    g2d = g.unsqueeze(1) * g.unsqueeze(0)
    return g2d.view(1, 1, size, size)


class LearnableEdgeDetector(nn.Module):
    """
    Learnable edge detection module.

    - Two learnable 3x3 kernels initialized as Sobel x / y (depthwise separable
      since they operate on a single input channel).
    - N directional edge detectors (learnable, initialized as rotated Sobel
      kernels) that adapt to dominant anatomical orientations during TTA.
    - Output: gradient magnitude + max directional response.
    """

    def __init__(self, num_directions=8):
        super().__init__()

        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]])

        self.conv_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv_x.weight.data = sobel_x.view(1, 1, 3, 3)
        self.conv_y.weight.data = sobel_y.view(1, 1, 3, 3)

        self.num_directions = num_directions
        self.dir_conv = nn.Conv2d(1, num_directions, 3, padding=1, bias=False)
        dir_init = torch.zeros(num_directions, 1, 3, 3)
        for i in range(num_directions):
            angle = i * np.pi / num_directions
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dir_init[i, 0] = cos_a * sobel_x + sin_a * sobel_y
        self.dir_conv.weight.data = dir_init

    def forward(self, x):
        """x: [1, 1, H, W]  ->  [1, 1, H, W]"""
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        dir_resp = self.dir_conv(x)                         # [1, N, H, W]
        dir_max = dir_resp.abs().max(dim=1, keepdim=True)[0]  # [1, 1, H, W]

        return grad_mag + dir_max


class StructureTensorResponse(nn.Module):
    """
    Structure Tensor for robust local structure analysis.

    Computes eigenvalues of the gradient structure tensor after Gaussian
    smoothing, then derives:
      coherence  = (λ1 - λ2) / (λ1 + λ2)   → anisotropy (0 = isotropic)
      energy     = λ1 / (λ1 + ε)            → normalised gradient energy

    The product coherence × energy highlights oriented structures (vessel
    walls, organ boundaries, tumour margins) while suppressing flat regions,
    isotropic noise, and isolated speckle — more stable than Laplacian for
    medical images with weak or infiltrative boundaries.
    """

    def __init__(self, sigma=1.5, kernel_size=7):
        super().__init__()
        gk = _gaussian_kernel_2d(kernel_size, sigma)
        self.register_buffer('gaussian', gk)
        self.pad = kernel_size // 2

    def forward(self, x):
        """x: [1, 1, H, W]  ->  [1, 1, H, W]"""
        sx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32, device=x.device)
        sy = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32, device=x.device)
        sx = sx.view(1, 1, 3, 3)
        sy = sy.view(1, 1, 3, 3)

        Ix = F.conv2d(x, sx, padding=1)
        Iy = F.conv2d(x, sy, padding=1)

        Sxx = Ix * Ix
        Syy = Iy * Iy
        Sxy = Ix * Iy

        Sxx = F.conv2d(Sxx, self.gaussian, padding=self.pad)
        Syy = F.conv2d(Syy, self.gaussian, padding=self.pad)
        Sxy = F.conv2d(Sxy, self.gaussian, padding=self.pad)

        trace = Sxx + Syy
        det = Sxx * Syy - Sxy * Sxy
        disc = torch.sqrt(torch.clamp(trace ** 2 - 4. * det, min=0.))
        lambda1 = (trace + disc) / 2.   # larger eigenvalue
        lambda2 = (trace - disc) / 2.   # smaller eigenvalue

        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
        energy = lambda1 / (lambda1 + 1.0)

        return coherence * energy


class LearnableBezierTransform(nn.Module):
    """
    Structure-aware SBCT with learnable edge kernels and Structure Tensor.

    Output: 3-channel image [3, H, W] from single-channel or 3-channel input.
    All parameters (Bezier control points + edge kernels) are trainable.
    """

    def __init__(self, num_directions=8):
        super().__init__()

        self.cp_intensity = nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0]))
        self.cp_edge      = nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0]))
        self.cp_structure = nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))

        self.edge_detector = LearnableEdgeDetector(num_directions)
        self.structure_tensor = StructureTensorResponse()

    def forward(self, x, base_name=None):
        """
        Args:
            x: [C, H, W], C ∈ {1, 3}, values in [0, 1].
            base_name: (unused) kept for API compatibility with callers.

        Returns:
            [3, H, W]
        """
        assert x.dim() == 3, f"Expected 3D [C,H,W], got {x.shape}"
        c = x.shape[0]
        assert c in [1, 3], f"Expected 1 or 3 channels, got {c}"

        cp1 = torch.sigmoid(self.cp_intensity)
        cp2 = torch.sigmoid(self.cp_edge)
        cp3 = torch.sigmoid(self.cp_structure)

        if c == 1:
            f_intensity = self.bezier_curve(cp1, x)

            x4 = x.unsqueeze(0)                              # [1, 1, H, W]
            edge = self.edge_detector(x4)                     # [1, 1, H, W]
            edge = self._normalize(edge).squeeze(0)           # [1, H, W]
            f_edge = self.bezier_curve(cp2, edge)

            st = self.structure_tensor(x4)                    # [1, 1, H, W]
            st = self._normalize(st).squeeze(0)               # [1, H, W]
            f_struct = self.bezier_curve(cp3, st)

            return torch.cat([f_intensity, f_edge, f_struct], dim=0)

        out_c0 = self.bezier_curve(cp1, x[0:1])
        out_c1 = self.bezier_curve(cp2, x[1:2])
        out_c2 = self.bezier_curve(cp3, x[2:3])
        return torch.cat([out_c0, out_c1, out_c2], dim=0)

    def bezier_curve(self, cp, x):
        P0, P1, P2, P3 = cp
        t = x
        omt = 1. - t
        f = (omt ** 3) * P0 + (3. * t * omt ** 2) * P1 \
            + (3. * t ** 2 * omt) * P2 + (t ** 3) * P3
        return torch.clamp(f, 0.0, 1.0)

    @staticmethod
    def _normalize(x, eps=1e-6):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)
