"""BUSI ultrasound dataset with perturbed-box CSV support."""
import os
import cv2
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


def collect_busi_samples(root_dir):
    """Scan benign/ and malignant/ subdirectories, pair images with masks. Excludes normal/."""
    samples = []
    for subdir in ['benign', 'malignant']:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        all_files = sorted(os.listdir(subdir_path))
        image_to_masks = {}
        for fname in all_files:
            full = os.path.join(subdir, fname)  # relative path from root_dir
            stem, ext = os.path.splitext(fname)
            if '_mask' in stem:
                if stem.endswith('_mask'):
                    base_stem = stem[:-5]
                else:
                    idx = stem.rfind('_mask')
                    base_stem = stem[:idx]
                image_to_masks.setdefault(base_stem, []).append(full)
            else:
                image_to_masks.setdefault(stem, [])

        for base_stem, mask_paths in image_to_masks.items():
            if not mask_paths:
                continue
            image_rel = os.path.join(subdir, base_stem + '.png')
            if not os.path.exists(os.path.join(root_dir, image_rel)):
                candidates = [f for f in all_files if f.startswith(base_stem) and '_mask' not in f]
                if candidates:
                    image_rel = os.path.join(subdir, candidates[0])
                else:
                    continue
            samples.append((image_rel, mask_paths))
    return samples


class BUSIDataset(Dataset):
    """Breast Ultrasound (BUSI) dataset loader for TTA.

    Scans benign/ and malignant/ subdirectories (excludes normal/).
    Multi-instance masks (_mask.png, _mask_1.png, ...) are merged into a single label map
    (values 1, 2, ...) then decoded into binary masks.
    """

    def __init__(self, cfg, root_dir, transform=None, if_self_training=False):
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
        self.samples = collect_busi_samples(root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_rel, mask_rels = self.samples[idx]
        image_path = os.path.join(self.root_dir, image_rel)
        basename = os.path.splitext(os.path.basename(image_rel))[0]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        combined_mask = None
        for label_val, mask_rel in enumerate(mask_rels, start=1):
            mask_path = os.path.join(self.root_dir, mask_rel)
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = (m > 127).astype(np.uint8)
            if combined_mask is None:
                combined_mask = np.zeros_like(m, dtype=np.uint8)
            combined_mask[m > 0] = label_val

        if combined_mask is None or combined_mask.sum() == 0:
            height, width = image.shape[:2]
            masks = [np.zeros((height, width), dtype=np.uint8)]
            bboxes = [[0, 0, width, height]]
            bboxes_coarse = [[0, 0, width, height]]
            categories = ['0']
        else:
            height, width = combined_mask.shape
            gt_masks = decode_mask(torch.tensor(combined_mask[None, :, :])).numpy().astype(np.uint8)

            offset_factor = self.box_offset if self.box_offset is not None else 0.2
            csv_boxes = self.box_map.get(basename) if self.box_map is not None else None
            masks, bboxes, bboxes_coarse, categories = [], [], [], []
            inst_idx = 0
            for mask in gt_masks:
                non_zero = cv2.findNonZero(mask)
                if non_zero is None:
                    continue
                x, y, w, h = cv2.boundingRect(non_zero)

                if csv_boxes is not None and inst_idx < len(csv_boxes):
                    x1, y1, x2, y2 = csv_boxes[inst_idx]
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
                inst_idx += 1
                x1c, y1c, x2c, y2c = x1, y1, x2, y2

                masks.append(mask)
                bboxes.append([x1,  y1,  x2,  y2])
                bboxes_coarse.append([x1c, y1c, x2c, y2c])
                categories.append('0')

            if len(masks) == 0:
                masks = [combined_mask]
                bboxes = [[0, 0, width, height]]
                bboxes_coarse = [[0, 0, width, height]]
                categories = ['0']

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

    all_dataset = BUSIDataset(
        cfg,
        root_dir=root_dir,
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
