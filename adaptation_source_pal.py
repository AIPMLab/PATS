"""PAL prompt loop on a frozen segmenter (SAM or MedSAM, selected via --cfg)."""
import argparse
import importlib.util
import os
import time
import yaml

import numpy as np
import pandas as pd
import torch
import cv2
import lightning as L
from box import Box
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from configs.config import cfg as base_cfg_mod  # noqa: F401
from datasets import call_load_dataset
from model import Model
from utils.eval_utils import AverageMeter, get_prompts, calculate_metrics, LAST_EVAL
from utils.tools import create_csv


def expand_bbox(x, y, w, h, width, height, expand_ratio=0.0):
    new_x = max(0, x)
    new_y = max(0, y)
    new_x2 = min(width, x + w)
    new_y2 = min(height, y + h)
    return new_x, new_y, new_x2, new_y2


def train_sam(cfg: Box, fabric: L.Fabric, model: Model, all_dataloader: DataLoader):
    dice_scores = AverageMeter()
    assd_scores = AverageMeter()
    hd95_scores = AverageMeter()
    case_results = []
    num_epochs = 1

    pal_K = int(getattr(cfg, 'pal_K', None) or 3)
    pal_thresh = getattr(cfg, 'pal_thresh', None)
    pal_thresh = 0.2 if pal_thresh is None else float(pal_thresh)

    for epoch in range(0, num_epochs):
        for iter, data in enumerate(all_dataloader):
            if getattr(cfg, 'limit', None) and iter >= int(cfg.limit):
                break
            images_test, bboxes_test, bboxes_test_coarse, gt_masks_test, basenames, \
                images_weak, images_strong, bboxes, gt_masks = data
            batch_size = images_test.size(0)

            prompts_init = get_prompts(cfg, bboxes_test, gt_masks_test)
            with torch.no_grad():
                _, masks0, ious0, _ = model(images_test, prompts_init)
            pal_prompts = [prompts_init]
            pal_masks = [masks0]
            pal_ious = [ious0]

            for k in range(1, pal_K):
                refined_bboxes_list = []
                for i in range(batch_size):
                    mask_2d = (pal_masks[-1][i][0] > 0).cpu().numpy().astype(np.uint8).squeeze()
                    iou_score = pal_ious[-1][i].mean().item()
                    non_zero_pts = cv2.findNonZero(mask_2d)
                    if iou_score > pal_thresh and non_zero_pts is not None:
                        x, y, w, h = cv2.boundingRect(non_zero_pts)
                        x1, y1, x2, y2 = expand_bbox(x, y, w, h, 1024, 1024)
                        refined_bboxes_list.append(torch.tensor(
                            [[x1, y1, x2, y2]], device=fabric.device, dtype=torch.float32))
                    else:
                        refined_bboxes_list.append(bboxes_test[i].to(fabric.device).float())
                prompts_k = get_prompts(cfg, refined_bboxes_list, gt_masks_test)
                with torch.no_grad():
                    _, masks_k, ious_k, _ = model(images_test, prompts_k)
                pal_prompts.append(prompts_k)
                pal_masks.append(masks_k)
                pal_ious.append(ious_k)

            pal_scores = [p[0].mean().item() for p in pal_ious]  # batch=1
            k_star = int(np.argmax(pal_scores))
            pred_masks_sel = pal_masks[k_star]
            pal_record = {'pal_ious': ';'.join(f'{s:.4f}' for s in pal_scores), 'pal_k': k_star}

            num_images = images_test.size(0)
            for pred_mask, gt_mask, basename in zip(pred_masks_sel, gt_masks_test, basenames):
                pred_mask = (pred_mask > 0.5).float()
                gt_mask = (gt_mask > 0.5).float()
                batch_dice, batch_assd, batch_hd95 = calculate_metrics(pred_mask, gt_mask)
                dice_scores.update(batch_dice, num_images)
                assd_scores.update(batch_assd, num_images)
                hd95_scores.update(batch_hd95, num_images)

                _pm = pred_mask.detach().cpu().numpy()
                while _pm.ndim > 2:
                    _pm = _pm.max(axis=0) if _pm.shape[0] > 1 else _pm[0]
                pred_mask_np = (_pm > 0.5).astype(np.uint8) * 255
                save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basename}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                from PIL import Image
                Image.fromarray(pred_mask_np).save(save_path)

                case_results.append({
                    "basename": basename,
                    "Dice": batch_dice,
                    "ASSD": batch_assd,
                    "HD95": batch_hd95,
                    "gt_empty": LAST_EVAL.get('gt_empty', -1),
                    "pred_empty": LAST_EVAL.get('pred_empty', -1),
                    **pal_record,
                })
                fabric.print(f"{basename} Dice: {batch_dice:.4f} - HD95: {batch_hd95:.4f}")
            torch.cuda.empty_cache()

    fabric.print(f'Test Ending...: Mean Dice: [{dice_scores.avg:.4f}] '
                 f'-- Mean HD95: [{hd95_scores.avg:.4f}]')
    case_results.append({
        "basename": "Average",
        "Dice": dice_scores.avg,
        "ASSD": assd_scores.avg,
        "HD95": hd95_scores.avg,
    })
    df = pd.DataFrame(case_results)
    if fabric.global_rank == 0:
        csv_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-test-results.csv")
        df.to_csv(csv_path, index=False)


def main(cfg: Box) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)
    fabric = L.Fabric(accelerator="auto", devices=num_devices, strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")])
    fabric.launch()
    _seed = getattr(cfg, 'seed', None)
    fabric.seed_everything((_seed if _seed is not None else 1337) + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        with open(os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml"), "w") as f:
            yaml.dump(cfg.to_dict(), f)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"),
                   csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()  # freeze everything; no LoRA, no updates

    load_datasets = call_load_dataset(cfg)
    all_data, _, _, _ = load_datasets(cfg, model.model.image_encoder.img_size)
    all_data = fabric._setup_dataloader(all_data)
    train_sam(cfg, fabric, model, all_data)


def load_config(cfg_path: str, args: argparse.Namespace):
    config_module = importlib.import_module(cfg_path.replace('.py', '').replace('/', '.'))
    cfg = Box(config_module.config)
    if hasattr(config_module, 'base_config'):
        cfg.merge_update(config_module.base_config)
    if args.dataset:
        cfg.dataset = args.dataset
    if args.prompt:
        cfg.prompt = args.prompt
    if args.gpu_ids:
        cfg.gpu_ids = args.gpu_ids
    if args.model_type:
        cfg.model.type = args.model_type
    if args.model_ckpt_dir:
        cfg.model.checkpoint = args.model_ckpt_dir
    name = 'sam_pal' if (getattr(cfg, 'model', {}).get('type', 'vit_b') == 'vit_b') else 'medsam_pal'
    cfg.out_dir = f"output/{name}_tta/" + args.dataset
    for _key in ('box_csv', 'box_offset', 'seed', 'pal_K', 'pal_thresh', 'limit'):
        if getattr(args, _key, None) is not None:
            cfg[_key] = getattr(args, _key)
    if getattr(args, 'out_dir', None):
        cfg.out_dir = args.out_dir
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-adaptation + PAL control")
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--gpu_ids', type=str)
    parser.add_argument('--model_type', type=str, default=None, choices=['vit_b', 'medsam'])
    parser.add_argument('--model_ckpt_dir', type=str, default=None)
    parser.add_argument('--box_csv', type=str, default=None)
    parser.add_argument('--box_offset', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pal_K', type=int, default=None)
    parser.add_argument('--pal_thresh', type=float, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
