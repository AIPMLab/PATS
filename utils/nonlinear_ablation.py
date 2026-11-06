"""Transform registry for the E-SBCT component ablation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _gaussian_kernel_2d(size=7, sigma=1.5):
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    g2d = g.unsqueeze(1) * g.unsqueeze(0)
    return g2d.view(1, 1, size, size)


def _normalize(x, eps=1e-6):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


class BaselineBezierTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.cp1 = nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0]))
        self.cp2 = nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0]))
        self.cp3 = nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))

    def forward(self, x, base_name=None):
        assert x.dim() == 3, f"Expected 3D [C,H,W], got {x.shape}"
        c = x.shape[0]
        assert c in [1, 3], f"Expected 1 or 3 channels, got {c}"

        cp1 = torch.sigmoid(self.cp1)
        cp2 = torch.sigmoid(self.cp2)
        cp3 = torch.sigmoid(self.cp3)

        if c == 1:
            f1 = self._bezier(cp1, x)
            f2 = self._bezier(cp2, x)
            f3 = self._bezier(cp3, x)
            return torch.cat([f1, f2, f3], dim=0)
        else:
            out0 = self._bezier(cp1, x[0:1])
            out1 = self._bezier(cp2, x[1:2])
            out2 = self._bezier(cp3, x[2:3])
            return torch.cat([out0, out1, out2], dim=0)

    def _bezier(self, cp, x):
        P0, P1, P2, P3 = cp
        t = x
        omt = 1. - t
        f = omt ** 3 * P0 + 3. * t * omt ** 2 * P1 \
            + 3. * t ** 2 * omt * P2 + t ** 3 * P3
        return torch.clamp(f, 0.0, 1.0)


class SobelBezierTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.cp1 = nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0]))
        self.cp2 = nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0]))
        self.cp3 = nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))

        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x, base_name=None):
        assert x.dim() == 3, f"Expected 3D [C,H,W], got {x.shape}"
        c = x.shape[0]
        assert c in [1, 3], f"Expected 1 or 3 channels, got {c}"

        cp1 = torch.sigmoid(self.cp1)
        cp2 = torch.sigmoid(self.cp2)
        cp3 = torch.sigmoid(self.cp3)

        if c == 1:
            f_intensity = self._bezier(cp1, x)

            x4 = x.unsqueeze(0)
            gx = F.conv2d(x4, self.sobel_x, padding=1)
            gy = F.conv2d(x4, self.sobel_y, padding=1)
            grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
            grad = _normalize(grad).squeeze(0)
            f_edge = self._bezier(cp2, grad)

            f3 = self._bezier(cp3, x)
            return torch.cat([f_intensity, f_edge, f3], dim=0)
        else:
            out0 = self._bezier(cp1, x[0:1])
            out1 = self._bezier(cp2, x[1:2])
            out2 = self._bezier(cp3, x[2:3])
            return torch.cat([out0, out1, out2], dim=0)

    def _bezier(self, cp, x):
        P0, P1, P2, P3 = cp
        t = x
        omt = 1. - t
        f = omt ** 3 * P0 + 3. * t * omt ** 2 * P1 \
            + 3. * t ** 2 * omt * P2 + t ** 3 * P3
        return torch.clamp(f, 0.0, 1.0)


class DirectionalBezierTransform(nn.Module):
    def __init__(self, num_directions=8):
        super().__init__()
        self.cp1 = nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0]))
        self.cp2 = nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0]))
        self.cp3 = nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))

        self.num_directions = num_directions
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32)

        self.dir_conv = nn.Conv2d(1, num_directions, 3, padding=1, bias=False)
        dir_init = torch.zeros(num_directions, 1, 3, 3)
        for i in range(num_directions):
            angle = i * np.pi / num_directions
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dir_init[i, 0] = cos_a * sobel_x + sin_a * sobel_y
        self.dir_conv.weight.data = dir_init

    def forward(self, x, base_name=None):
        assert x.dim() == 3, f"Expected 3D [C,H,W], got {x.shape}"
        c = x.shape[0]
        assert c in [1, 3], f"Expected 1 or 3 channels, got {c}"

        cp1 = torch.sigmoid(self.cp1)
        cp2 = torch.sigmoid(self.cp2)
        cp3 = torch.sigmoid(self.cp3)

        if c == 1:
            f_intensity = self._bezier(cp1, x)

            x4 = x.unsqueeze(0)
            dir_resp = self.dir_conv(x4)
            dir_max = dir_resp.abs().max(dim=1, keepdim=True)[0]
            dir_max = _normalize(dir_max).squeeze(0)
            f_edge = self._bezier(cp2, dir_max)

            f3 = self._bezier(cp3, x)
            return torch.cat([f_intensity, f_edge, f3], dim=0)
        else:
            out0 = self._bezier(cp1, x[0:1])
            out1 = self._bezier(cp2, x[1:2])
            out2 = self._bezier(cp3, x[2:3])
            return torch.cat([out0, out1, out2], dim=0)

    def _bezier(self, cp, x):
        P0, P1, P2, P3 = cp
        t = x
        omt = 1. - t
        f = omt ** 3 * P0 + 3. * t * omt ** 2 * P1 \
            + 3. * t ** 2 * omt * P2 + t ** 3 * P3
        return torch.clamp(f, 0.0, 1.0)


class StructureTensorBezierTransform(nn.Module):
    def __init__(self, sigma=1.5, kernel_size=7):
        super().__init__()
        self.cp1 = nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0]))
        self.cp2 = nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0]))
        self.cp3 = nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))

        gk = _gaussian_kernel_2d(kernel_size, sigma)
        self.register_buffer('gaussian', gk)
        self.pad = kernel_size // 2

        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32)
        self.register_buffer('sx', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sy', sobel_y.view(1, 1, 3, 3))

    def forward(self, x, base_name=None):
        assert x.dim() == 3, f"Expected 3D [C,H,W], got {x.shape}"
        c = x.shape[0]
        assert c in [1, 3], f"Expected 1 or 3 channels, got {c}"

        cp1 = torch.sigmoid(self.cp1)
        cp2 = torch.sigmoid(self.cp2)
        cp3 = torch.sigmoid(self.cp3)

        if c == 1:
            f_intensity = self._bezier(cp1, x)
            f2 = self._bezier(cp2, x)

            x4 = x.unsqueeze(0)
            Ix = F.conv2d(x4, self.sx, padding=1)
            Iy = F.conv2d(x4, self.sy, padding=1)

            Sxx = Ix * Ix
            Syy = Iy * Iy
            Sxy = Ix * Iy

            Sxx = F.conv2d(Sxx, self.gaussian, padding=self.pad)
            Syy = F.conv2d(Syy, self.gaussian, padding=self.pad)
            Sxy = F.conv2d(Sxy, self.gaussian, padding=self.pad)

            trace = Sxx + Syy
            det = Sxx * Syy - Sxy * Sxy
            disc = torch.sqrt(torch.clamp(trace ** 2 - 4. * det, min=0.))
            lambda1 = (trace + disc) / 2.
            lambda2 = (trace - disc) / 2.

            coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
            energy = lambda1 / (lambda1 + 1.0)
            st = _normalize(coherence * energy).squeeze(0)
            f_struct = self._bezier(cp3, st)

            return torch.cat([f_intensity, f2, f_struct], dim=0)
        else:
            out0 = self._bezier(cp1, x[0:1])
            out1 = self._bezier(cp2, x[1:2])
            out2 = self._bezier(cp3, x[2:3])
            return torch.cat([out0, out1, out2], dim=0)

    def _bezier(self, cp, x):
        P0, P1, P2, P3 = cp
        t = x
        omt = 1. - t
        f = omt ** 3 * P0 + 3. * t * omt ** 2 * P1 \
            + 3. * t ** 2 * omt * P2 + t ** 3 * P3
        return torch.clamp(f, 0.0, 1.0)


TRANSFORM_REGISTRY = {
    'baseline':    BaselineBezierTransform,
    'sobel':       SobelBezierTransform,
    'directional': DirectionalBezierTransform,
    'structure':   StructureTensorBezierTransform,
}


def get_transform(name: str):
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown ablation '{name}'. Choose from: {list(TRANSFORM_REGISTRY.keys())}")
    return TRANSFORM_REGISTRY[name]()
