"""Multi-center Prostate MRI dataset (HDF5) with perturbed-box CSV support."""
import os
import cv2
import h5py
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from datasets.tools import ResizeAndPad, soft_transform, collate_fn, decode_mask


def expand_bbox(x, y, w, h, width, height, expand_ratio=0.0):
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)
    new_x2 = min(width, x + w + (new_w - w) / 2)
    new_y2 = min(height, y + h + (new_h - h) / 2)
    return new_x, new_y, new_x2, new_y2


class ProstateDataset(Dataset):
    """
    Loads Prostate H5 slices for test-time adaptation.
    Each H5 file has shape (N, H, W) for 'image' and 'label'.
    all_slices.list contains entries like 'Case00_slice_5'.
    Returns 9-tuple for self-training (if_self_training=True) or 4-tuple for inference.
    """

    def __init__(self, cfg, root_dir, slice_list_file, transform=None, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.if_self_training = if_self_training
        self.box_offset = getattr(cfg, 'box_offset', None)
        box_csv = getattr(cfg, 'box_csv', None)
        self.box_map = None
        if box_csv:
            bdf = pd.read_csv(box_csv)
            self.box_map = {
                k: g.sort_values('inst')[['x1', 'y1', 'x2', 'y2']].values.tolist()
                for k, g in bdf.groupby('basename')
            }

        with open(slice_list_file, 'r') as f:
            self.slice_names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.slice_names)

    def __getitem__(self, idx):
        slice_name = self.slice_names[idx]  # e.g. 'Case00_slice_5'
        parts = slice_name.rsplit('_slice_', 1)
        case_name = parts[0]          # 'Case00'
        slice_idx = int(parts[1])     # 5

        h5_path = os.path.join(self.root_dir, 'data', f'{case_name}.h5')
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][slice_idx]   # (H, W), float32
            label = f['label'][slice_idx]   # (H, W), int8

        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)
        image = (image * 255).astype(np.uint8)
        image = np.stack([image, image, image], axis=-1)  # (H, W, 3)

        gt_mask = (label > 0).astype(np.uint8)  # binary mask
        height, width = gt_mask.shape

        if gt_mask.sum() == 0:
            masks = [gt_mask]
            bboxes = [[0, 0, width, height]]
            bboxes_coarse = [[0, 0, width, height]]
            categories = ['0']
        else:
            gt_masks_decoded = decode_mask(
                torch.tensor(gt_mask[None, :, :].astype(np.int64))
            ).numpy().astype(np.uint8)

            offset_factor = self.box_offset if self.box_offset is not None else 0.25
            csv_boxes = self.box_map.get(slice_name) if self.box_map is not None else None

            masks, bboxes, bboxes_coarse, categories = [], [], [], []
            for inst_idx, mask in enumerate(gt_masks_decoded):
                non_zero = cv2.findNonZero(mask)
                if non_zero is None:
                    continue
                x, y, w, h = cv2.boundingRect(non_zero)

                if csv_boxes is not None and inst_idx < len(csv_boxes):
                    bx1, by1, bx2, by2 = csv_boxes[inst_idx]
                    x1, y1, x2, y2 = bx1, by1, bx2, by2
                else:
                    max_offset_x = int(w * offset_factor)
                    max_offset_y = int(h * offset_factor)

                    offset_x = random.randint(-max_offset_x, max_offset_x)
                    offset_y = random.randint(-max_offset_y, max_offset_y)
                    offset_w = random.randint(-max_offset_x, max_offset_x)
                    offset_h = random.randint(-max_offset_y, max_offset_y)

                    x_p = max(0, min(width,  x + offset_x))
                    y_p = max(0, min(height, y + offset_y))
                    w_p = max(1, min(width  - x_p, w + offset_w))
                    h_p = max(1, min(height - y_p, h + offset_h))

                    x1,  y1,  x2,  y2  = expand_bbox(x_p, y_p, w_p, h_p, width, height, expand_ratio=0.0)
                x1c, y1c, x2c, y2c = x1, y1, x2, y2

                masks.append(mask)
                bboxes.append([x1,  y1,  x2,  y2])
                bboxes_coarse.append([x1c, y1c, x2c, y2c])
                categories.append('0')

            if len(masks) == 0:
                masks = [gt_mask]
                bboxes = [[0, 0, width, height]]
                bboxes_coarse = [[0, 0, width, height]]
                categories = ['0']

        basename = slice_name

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(
                image, bboxes, masks, categories
            )
            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(
                    image_weak, masks_weak, np.array(bboxes_weak)
                )
                image_strong = self.transform.transform_image(image_strong)
                image_coarse = image
                masks_coarse = masks
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
                _, _, bboxes_coarse = self.transform(
                    image_coarse, masks_coarse, np.array(bboxes_coarse)
                )

            bboxes = np.stack(bboxes, axis=0)
            bboxes_coarse = np.stack(bboxes_coarse, axis=0)
            masks = np.stack(masks, axis=0)
            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)

            return (
                image,
                torch.tensor(bboxes),
                torch.tensor(bboxes_coarse),
                torch.tensor(masks).float(),
                basename,
                image_weak,
                image_strong,
                torch.tensor(bboxes_weak),
                torch.tensor(masks_weak).float(),
            )
        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float(), basename


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    dataset_name = cfg.dataset
    root_dir = cfg.datasets[dataset_name].root_dir
    slice_list = cfg.datasets[dataset_name].all_list

    all_dataset = ProstateDataset(
        cfg,
        root_dir=root_dir,
        slice_list_file=slice_list,
        transform=transform,
        if_self_training=cfg.augment,
    )
    all_dataloader = DataLoader(
        all_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return all_dataloader, all_dataloader, all_dataloader, all_dataloader
