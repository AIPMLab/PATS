"""Pre-generate perturbed prompt boxes so every method shares the same boxes
per (dataset, offset, seed). Replicates the in-loader perturbation:
    x,y,w,h = cv2.boundingRect(mask); m = int(w*off) / int(h*off)
    ox,oy,ow,oh ~ Uniform{-m..m} (python random.randint, inclusive)
    clamp, then tight box (expand_ratio=0).
Output CSV columns:
    basename,inst,x1,y1,x2,y2,gt_x1,gt_y1,gt_x2,gt_y2,gt_empty
Empty-GT slices get a single full-image box row (gt_empty=1), matching the
loaders' fallback.
"""
import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.tools import decode_mask  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def instances_from_mask(mask2d):
    """mask2d: HxW np.uint8 (binary or multi-label). Returns list of binary masks."""
    if mask2d.sum() == 0:
        return []
    dec = decode_mask(torch.tensor(mask2d[None, :, :].astype(np.int64))).numpy().astype(np.uint8)
    return [m for m in dec if m.sum() > 0]


def tight_box(mask, width, height):
    x, y, w, h = cv2.boundingRect(mask)
    return [x, y, x + w, y + h]


def perturb_box(gt_box, width, height, offset, rng):
    x1, y1, x2, y2 = gt_box
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    max_ox = int(w * offset)
    max_oy = int(h * offset)
    ox = rng.randint(-max_ox, max_ox)
    oy = rng.randint(-max_oy, max_oy)
    ow = rng.randint(-max_ox, max_ox)
    oh = rng.randint(-max_oy, max_oy)
    x_p = max(0, min(width, x + ox))
    y_p = max(0, min(height, y + oy))
    w_p = max(1, min(width - x_p, w + ow))
    h_p = max(1, min(height - y_p, h + oh))
    return [x_p, y_p, x_p + w_p, y_p + h_p]


def slices_brats(name):
    df = pd.read_csv(os.path.join(ROOT, 'data', name, 'train.csv'), encoding='gbk')
    for img_rel, lbl_rel in zip(df.iloc[:, 0], df.iloc[:, 1]):
        base = os.path.basename(img_rel).split('.')[0]
        lbl = os.path.join(ROOT, 'data', name, lbl_rel)
        yield base, ('nii', lbl)


def slices_h5(name, list_file='all_slices.list'):
    import h5py
    with open(os.path.join(ROOT, 'data', name, list_file)) as f:
        names = [l.strip() for l in f if l.strip()]
    h5_cache = {}
    for n in names:
        case, idx = n.rsplit('_slice_', 1)
        idx = int(idx)
        if name == 'ACDC':
            idx = idx - 1  # list is 1-based for ACDC
        if case not in h5_cache:
            h5_cache[case] = h5py.File(os.path.join(ROOT, 'data', name, 'data', f'{case}.h5'), 'r')
        yield n, ('h5', h5_cache[case], idx)


def slices_busi():
    from datasets.BUSI import collect_busi_samples
    root = os.path.join(ROOT, 'data', 'Dataset_BUSI', 'Dataset_BUSI_with_GT')
    for image_rel, mask_rels in collect_busi_samples(root):
        base = os.path.splitext(os.path.basename(image_rel))[0]
        yield base, ('busi', root, mask_rels)


def load_gt(kind_obj):
    """Return (mask2d uint8, width, height) or (None, w, h) if GT empty."""
    kind = kind_obj[0]
    import nibabel as nib
    if kind == 'nii':
        m = nib.load(kind_obj[1]).get_fdata()
        if m.ndim == 3:
            m = m[0, :, :]
        m = (m > 0).astype(np.uint8) if m.max() <= 1 else (m * 255).astype(np.uint8)
        m = (m > 0).astype(np.uint8)
        return m
    if kind == 'h5':
        f, idx = kind_obj[1], kind_obj[2]
        lab = f['label'][idx]
        return (lab > 0).astype(np.uint8) if lab.max() <= 1 else lab.astype(np.uint8)
    if kind == 'busi':
        _, root, mask_rels = kind_obj
        combined = None
        for label_val, rel in enumerate(mask_rels, start=1):
            m = cv2.imread(os.path.join(root, rel), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = (m > 127).astype(np.uint8)
            if combined is None:
                combined = np.zeros_like(m, dtype=np.uint8)
            combined[m > 0] = label_val
        if combined is None:
            return None
        return combined
    raise ValueError(kind)


DATASET_SLICES = {
    'BraTS_SSA_t2w_2D': lambda: slices_brats('BraTS_SSA_t2w_2D'),
    'BraTS_SSA_t2f_2D': lambda: slices_brats('BraTS_SSA_t2f_2D'),
    'BraTS_PED_t2w_2D': lambda: slices_brats('BraTS_PED_t2w_2D'),
    'BraTS_PED_t2f_2D': lambda: slices_brats('BraTS_PED_t2f_2D'),
    'Prostate': lambda: slices_h5('Prostate'),
    'ACDC': lambda: slices_h5('ACDC'),
    'BUSI': slices_busi,
}


def gen(dataset, offset, seed, out_dir):
    rng = random.Random(f'{dataset}|{offset}|{seed}')
    rows = []
    for base, kind_obj in DATASET_SLICES[dataset]():
        mask = load_gt(kind_obj)
        if mask is None or mask.sum() == 0:
            h, w = mask.shape if mask is not None else (0, 0)
            if mask is None:  # BUSI with no mask files: read image size
                rows.append([base, 0, -1, -1, -1, -1, -1, -1, -1, -1, 1])
                continue
            rows.append([base, 0, 0, 0, w, h, 0, 0, w, h, 1])
            continue
        h, w = mask.shape
        for i, inst in enumerate(instances_from_mask(mask)):
            gb = tight_box(inst, w, h)
            pb = perturb_box(gb, w, h, offset, rng)
            rows.append([base, i, *pb, *gb, 0])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{dataset}_off{int(round(offset*100))}_seed{seed}.csv')
    with open(out, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['basename', 'inst', 'x1', 'y1', 'x2', 'y2',
                     'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'gt_empty'])
        wr.writerows(rows)
    n_empty = sum(1 for r in rows if r[10] == 1)
    print(f'{out}  slices={len({r[0] for r in rows})} instances={len(rows)} emptyGT={n_empty}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASET_SLICES.keys()))
    ap.add_argument('--offsets', nargs='+', type=float, default=[0.15, 0.20, 0.25])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1337, 1338, 1339])
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'boxes'))
    args = ap.parse_args()
    for ds in args.datasets:
        for off in args.offsets:
            for seed in args.seeds:
                gen(ds, off, seed, args.out_dir)


if __name__ == '__main__':
    main()
