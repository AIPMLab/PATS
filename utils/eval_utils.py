"""Validation loop, metrics (Dice/HD95), and prompt utilities."""
import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv
from medpy import metric
import numpy as np
from typing import Tuple
from PIL import Image
import pandas as pd


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


LAST_EVAL = {'pred_empty': -1, 'gt_empty': -1}


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse" or 'box' in cfg.prompt:
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts


    

def _collapse_to_2d(mask_np):
    """Collapse leading instance/batch dims of a boolean mask to 2D by
    any-instance merge (multi-lesion slices, e.g. BUSI, multi-class ACDC)."""
    import numpy as _np
    m = _np.asarray(mask_np).astype(bool)
    while m.ndim > 2:
        m = m.any(axis=0) if m.shape[0] > 1 else m[0]
    return m


def calculate_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    """v2 (journal revision): raw, uncapped metrics with explicit empty handling.

    Dice:    dc() when both non-empty; 1.0 both empty; 0.0 exactly one empty.
    HD95:    medpy hd95 UNCAPPED when both non-empty; grid diagonal when exactly
             one empty; 0.0 when both empty. Legacy cap-50 can be restored by
             setting env JR_HD95_CAP=50 (writes min(raw, 50) for comparability).
    ASSD:    real average surface distance (legacy code hardcoded 50).
    Multi-instance masks (leading dims > 1) are merged by any-instance before
    computing metrics. Empty-case flags land in LAST_EVAL.
    """
    import os as _os
    pred2d = _collapse_to_2d(pred_mask.cpu().numpy() > 0.5)
    gt2d = _collapse_to_2d(gt_mask.cpu().numpy() > 0.5)
    if pred2d.shape != gt2d.shape:
        raise ValueError(f'shape mismatch {pred2d.shape} vs {gt2d.shape}')

    pred_empty = bool(pred2d.sum() == 0)
    gt_empty = bool(gt2d.sum() == 0)
    LAST_EVAL['pred_empty'] = int(pred_empty)
    LAST_EVAL['gt_empty'] = int(gt_empty)
    diag = float(np.hypot(*gt2d.shape[-2:]))

    if pred_empty and gt_empty:
        batch_dice, batch_asd, batch_hd95 = 1.0, 0.0, 0.0
    elif pred_empty or gt_empty:
        batch_dice, batch_asd, batch_hd95 = 0.0, diag, diag
    else:
        p = pred2d.astype(np.uint8)
        g = gt2d.astype(np.uint8)
        batch_dice = metric.binary.dc(p, g)
        batch_hd95 = float(metric.binary.hd95(p, g))
        batch_asd = float(metric.binary.asd(p, g))

    cap = _os.environ.get('JR_HD95_CAP')
    if cap:
        batch_hd95 = min(batch_hd95, float(cap))

    return batch_dice, batch_asd, batch_hd95
def test_med(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    model.eval()
    dice_scores = AverageMeter()
    assd_scores = AverageMeter() 
    hd95_scores = AverageMeter() 

    case_results = []

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, basenames = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)
            _, pred_masks, _, _ = model(images, prompts)

            for pred_mask, gt_mask, basename in zip(pred_masks, gt_masks, basenames):
                pred_mask = (pred_mask > 0.5).float()
                gt_mask = (gt_mask > 0.5).float()

                batch_dice, batch_assd, batch_hd95 = calculate_metrics(pred_mask, gt_mask)

                dice_scores.update(batch_dice, num_images)
                assd_scores.update(batch_assd, num_images)
                hd95_scores.update(batch_hd95, num_images)

                pred_mask_np = pred_mask.squeeze().cpu().numpy()  
                pred_mask_np = (pred_mask_np * 255).astype(np.uint8)  

                save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basename}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                Image.fromarray(pred_mask_np).save(save_path)

                case_results.append({
                    "basename": basename, 
                    "Dice": batch_dice, 
                    "ASSD": batch_assd, 
                    "HD95": batch_hd95
                })

                fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")


            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean Dice: [{dice_scores.avg:.4f}] -- Mean ASSD: [{assd_scores.avg:.4f}] -- Mean HD95: [{hd95_scores.avg:.4f}]')

    case_results.append({
        "basename": "Average",  
        "Dice": dice_scores.avg,
        "ASSD": assd_scores.avg,
        "HD95": hd95_scores.avg
    })

    df = pd.DataFrame(case_results)

    if fabric.global_rank == 0:
        csv_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-test-results.csv")
        df.to_csv(csv_path, index=False)

    model.train()
    return dice_scores.avg, assd_scores.avg, hd95_scores.avg

def validate_med(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    model.eval()
    dice_scores = AverageMeter() 

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, _ = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                pred_mask = (pred_mask > 0.5).cpu().numpy()  
                gt_mask = (gt_mask > 0.5).cpu().numpy()  

                batch_dice = metric.binary.dc(pred_mask, gt_mask)
                dice_scores.update(batch_dice, num_images)

            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean Dice: [{dice_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean Dice: [{dice_scores.avg:.4f}]')
    
    csv_dict = {
        "Name": name, 
        "Prompt": cfg.prompt, 
        "Mean Dice": f"{dice_scores.avg:.4f}", 
        "iters": iters
    }

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_dict, csv_head=cfg.csv_keys)

    model.train()
    return dice_scores.avg


def validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks,_ = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "iters": iters}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg


def unspervised_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    init_prompt = cfg.prompt
    cfg.prompt = "box"
    iou_box, f1_box = validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = "point"
    iou_point, f1_point = validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = init_prompt
    return iou_box, f1_box, iou_point, f1_point


def contrast_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0, loss: float = 0.):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "iters": iters, "loss": loss}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"metrics-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg
