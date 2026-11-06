"""Bezier input transform (image-only variant used by the SAM-TTA baseline)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import comb
from PIL import Image

class LearnableBezierTransform(nn.Module):
    def __init__(self, num_control_points=4):
        super(LearnableBezierTransform, self).__init__()
        self.num_control_points = num_control_points
        
        self.control_points_1 = nn.Parameter(torch.tensor([
            0.0,
            0.33,
            0.66,
            1.0
        ], dtype=torch.float32))
        
        self.control_points_2 = nn.Parameter(torch.tensor([
            0.0,
            0.2,
            0.8,
            1.0
        ], dtype=torch.float32))

        self.control_points_3 = nn.Parameter(torch.tensor([
            0.0,
            0.25,
            0.75,
            1.0
        ], dtype=torch.float32))
        self.control_points_11 = nn.Parameter(torch.tensor([
            0.0,
            0.33,
            0.66,
            1.0
        ], dtype=torch.float32))
        
        self.control_points_22 = nn.Parameter(torch.tensor([
            0.0,
            0.2,
            0.8,
            1.0
        ], dtype=torch.float32))

        self.control_points_33 = nn.Parameter(torch.tensor([
            0.0,
            0.25,
            0.75,
            1.0
        ], dtype=torch.float32))
        
    def forward_1(self, x):
        assert x.dim() == 3 and x.size(0) == 1, "expected input of shape [1, H, W]"
        
        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]

        f1 = self.bezier_curve(cp1, x)  # [1, H, W]
        f2 = self.bezier_curve(cp2, x)  # [1, H, W]
        f3 = self.bezier_curve(cp3, x)  # [1, H, W]
        
        output = torch.cat([f1, f2, f3], dim=0)
        
        self.save_image(x[0], f1[0], f2[0], f3[0], 'transformed')
        
        return output  # [3, H, W]
    
    def forward(self, x, base_name):
        assert x.dim() == 3, "expected a 3D tensor of shape [C, H, W]"
        c, h, w = x.shape
        assert c in [1, 3], "only 1-channel [1, H, W] or 3-channel [3, H, W] inputs are supported"

        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]

        if c == 1:
            f1 = self.bezier_curve(cp1, x)  # [1, H, W]
            f2 = self.bezier_curve(cp2, x)  # [1, H, W]
            f3 = self.bezier_curve(cp3, x)  # [1, H, W]
            
            output = torch.cat([f1, f2, f3], dim=0)  # [3, H, W]

            return output
        
        else:  # c == 3
            out_c0 = self.bezier_curve(cp1, x[0:1])  # [1, H, W]
            out_c1 = self.bezier_curve(cp2, x[1:2])  # [1, H, W]
            out_c2 = self.bezier_curve(cp3, x[2:3])  # [1, H, W]
            
            output = torch.cat([out_c0, out_c1, out_c2], dim=0)  # [3, H, W]

            return output
    def forward_2(self, x):
        assert x.dim() == 3, "expected a 3D tensor of shape [C, H, W]"
        c, h, w = x.shape
        assert c in [1, 3], "only 1-channel [1, H, W] or 3-channel [3, H, W] inputs are supported"

        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]
        cp11 = torch.sigmoid(self.control_points_11)  # [4]
        cp22 = torch.sigmoid(self.control_points_22)  # [4]
        cp33 = torch.sigmoid(self.control_points_33)  # [4]

        if c == 1:
            f1 = self.bezier_curve(cp1, x)  # [1, H, W]
            f2 = self.bezier_curve(cp2, x)  # [1, H, W]
            f3 = self.bezier_curve(cp3, x)  # [1, H, W]
            f11 = self.bezier_curve(cp11, x)  # [1, H, W]
            f22 = self.bezier_curve(cp22, x)  # [1, H, W]
            f33 = self.bezier_curve(cp33, x)  # [1, H, W]
            
            output = torch.cat([f1, f2, f3], dim=0)  # [3, H, W]
            output_2 = torch.cat([f11, f22, f33], dim=0)  # [3, H, W]

            return output, output_2
        
        else:  # c == 3
            out_c0 = self.bezier_curve(cp1, x[0:1])  # [1, H, W]
            out_c1 = self.bezier_curve(cp2, x[1:2])  # [1, H, W]
            out_c2 = self.bezier_curve(cp3, x[2:3])  # [1, H, W]
            f11 = self.bezier_curve(cp11, x[0:1])  # [1, H, W]
            f22 = self.bezier_curve(cp22, x[1:2])  # [1, H, W]
            f33 = self.bezier_curve(cp33, x[2:3])  # [1, H, W]
            
            
            output = torch.cat([out_c0, out_c1, out_c2], dim=0)  # [3, H, W]
            output_2 = torch.cat([f11, f22, f33], dim=0)  # [3, H, W]

            return output, output_2
    
    
    def bezier_curve(self, control_points, x):
        P0, P1, P2, P3 = control_points
        
        t = x                 # [1, H, W]
        one_minus_t = 1 - t   # [1, H, W]
        
        B0 = comb(3, 0) * (one_minus_t ** 3)      # [1, H, W]
        B1 = comb(3, 1) * (t ** 1) * (one_minus_t ** 2)
        B2 = comb(3, 2) * (t ** 2) * (one_minus_t ** 1)
        B3 = comb(3, 3) * (t ** 3) * (one_minus_t ** 0)
        
        f_t = B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3
        
        f_t = torch.clamp(f_t, 0.0, 1.0)
        
        return f_t  # [1, H, W]
    
    def save_image(self, img, img1, img2, img3, filename):
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        img3_np = img3.detach().cpu().numpy()
        img_np = img.detach().cpu().numpy()

        img1_np = self.normalize_image(img1_np)
        img2_np = self.normalize_image(img2_np)
        img3_np = self.normalize_image(img3_np)
        img_np = self.normalize_image(img_np)

        combined = np.concatenate([img_np, img1_np, img2_np, img3_np], axis=1)

        image = Image.fromarray(combined, mode='L')
        image.save(f"/home/data/SAM/wesam/show_4img/{filename}.png")
        print(f"Image saved as '{filename}.png'")

    @staticmethod
    def normalize_image(img):
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        return (img * 255).astype(np.uint8)

if __name__ == "__main__":
    height, width = 1024, 1024

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = LearnableBezierTransform().to(device)

    input_image = torch.rand(1, height, width, device=device)  # [1, 1024, 1024]

    transformed_image = transform(input_image)
    print(f"Transformed image shape: {transformed_image.shape}, device: {transformed_image.device}")
