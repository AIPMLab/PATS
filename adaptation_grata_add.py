"""GraTa with the PATS wrapper."""

import argparse
import csv
import importlib.util
import os
import time
import random

import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from box import Box
from PIL import Image
from torch.utils.data import DataLoader

from lightning.fabric.loggers import TensorBoardLogger

from datasets import call_load_dataset
from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import (AverageMeter, get_prompts,
                              calculate_metrics, LAST_EVAL)
from utils.tools import reduce_instances
from utils.nonlinear_learned import LearnableBezierTransform



def expand_bbox(x, y, w, h, width, height, expand_ratio=0.25):
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)
    new_x2 = min(width, x + w + (new_w - w) / 2)
    new_y2 = min(height, y + h + (new_h - h) / 2)
    return new_x, new_y, new_x2, new_y2



class GraTaSAM:
    """
    Gradient Alignment TTA optimizer for SAM.

    At each batch the optimizer:
      1. Computes grad of aux loss (entropy) on the current params.
      2. Perturbs params:  θ ← θ − ∇L_aux
      3. Computes grad of pseudo loss (consistency) at the perturbed point.
      4. Measures cosine similarity between ∇L_aux and ∇L_pse.
      5. Restores original params θ.
      6. Scales lr by ¼(cos+1)² and steps the base optimizer along ∇L_pse.
    """

    def __init__(self, params, base_optimizer, model, device):
        self.base_optimizer = base_optimizer
        self.model = model
        self.device = device
        self.init_lr = base_optimizer.param_groups[0]['lr']
        self.perturb_eps = 1e-12
        self.param_list = list(params)

        self._grata_state = {}   # p.id -> {"old_p": ..., "aux_g": ...}


    @torch.no_grad()
    def _save_and_perturb(self):
        """Save param copies and aux grads, then w ← w − g_aux."""
        for p in self.param_list:
            if p.grad is None:
                continue
            self._grata_state[id(p)] = {
                'old_p': p.data.clone(),
                'aux_g': p.grad.data.clone(),
            }
            p.data.sub_(p.grad)

    @torch.no_grad()
    def _restore(self):
        """Restore params to their pre-perturbation values."""
        for p in self.param_list:
            entry = self._grata_state.get(id(p))
            if entry is not None:
                p.data.copy_(entry['old_p'])


    @torch.no_grad()
    def _cosine(self):
        """Cosine similarity between the saved aux grad and current grad."""
        dot = 0.0
        norm_aux = 0.0
        norm_pse = 0.0
        for p in self.param_list:
            if p.grad is None:
                continue
            entry = self._grata_state.get(id(p))
            if entry is None:
                continue
            g_aux = entry['aux_g']
            g_pse = p.grad.data
            dot += (g_aux * g_pse).sum()
            norm_aux += (g_aux ** 2).sum()
            norm_pse += (g_pse ** 2).sum()
        eps = self.perturb_eps
        cos = dot / (torch.sqrt(norm_aux) * torch.sqrt(norm_pse) + eps)
        return cos.detach()


    def _entropy_loss(self, images, prompts):
        """
        Entropy minimisation — auxiliary loss.

        H = −Σ p·log p + (1−p)·log(1−p)
        Minimising H pushes predictions toward 0 or 1 (confident).
        """
        _, pred_masks, _, _ = self.model(images, prompts)
        loss = torch.tensor(0., device=self.device)
        n = 0
        for mask in pred_masks:
            prob = torch.sigmoid(mask)
            ent = -(prob * torch.log(prob + 1e-6)
                    + (1. - prob) * torch.log(1. - prob + 1e-6))
            loss = loss + ent.mean()
            n += 1
        return loss / max(n, 1)

    def _consistency_loss(self, images, prompts):
        """
        Consistency regularisation — pseudo loss.

        Pseudo-label = prediction on original image (no grad).
        Target      = prediction on style-augmented image.
        Loss        = BCE(target_logits, pseudo_label).
        """
        with torch.no_grad():
            _, masks_orig, _, _ = self.model(images, prompts)

        aug_images = self._style_augment(images)
        _, masks_aug, _, _ = self.model(aug_images, prompts)

        loss = torch.tensor(0., device=self.device)
        n = 0
        for m_aug, m_orig in zip(masks_aug, masks_orig):
            target = torch.sigmoid(m_orig.detach())
            loss = loss + F.binary_cross_entropy_with_logits(m_aug, target)
            n += 1
        return loss / max(n, 1)


    @staticmethod
    def _style_augment(images):
        """
        Apply brightness, contrast, gamma, and light Gaussian noise
        using only non-inplace ops to avoid autograd version conflicts.
        """
        aug = images
        B, C = aug.shape[0], aug.shape[1]

        for i in range(B):
            if random.random() < 0.75:
                aug = torch.cat(
                    [aug[:i],
                     aug[i:i + 1] * random.uniform(0.5, 1.5),
                     aug[i + 1:]]) if i < B - 1 else torch.cat(
                    [aug[:i], aug[i:i + 1] * random.uniform(0.5, 1.5)])

            if random.random() < 0.75:
                chans = []
                for c in range(C):
                    mu = aug[i, c].mean()
                    contrast = random.uniform(0.5, 1.5)
                    chans.append(
                        ((aug[i, c] - mu) * contrast + mu).unsqueeze(0))
                new_img = torch.cat(chans, dim=0).unsqueeze(0)
                aug = torch.cat(
                    [aug[:i], new_img, aug[i + 1:]]) if i < B - 1 else torch.cat(
                    [aug[:i], new_img])

            if random.random() < 0.75:
                gamma = random.uniform(0.5, 2.0)
                aug = torch.cat(
                    [aug[:i],
                     aug[i:i + 1].clamp(min=1e-8).pow(gamma),
                     aug[i + 1:]]) if i < B - 1 else torch.cat(
                    [aug[:i], aug[i:i + 1].clamp(min=1e-8).pow(gamma)])

            if random.random() < 0.5:
                aug = torch.cat(
                    [aug[:i],
                     aug[i:i + 1] + torch.randn_like(aug[i:i + 1]) * 0.02,
                     aug[i + 1:]]) if i < B - 1 else torch.cat(
                    [aug[:i],
                     aug[i:i + 1] + torch.randn_like(aug[i:i + 1]) * 0.02])

        return aug.clamp(0., 1.)


    def step(self, images, prompts):
        """
        Single GraTa update.

        Returns (aux_loss, pse_loss, cosine) as Python floats for logging.
        """
        self.base_optimizer.zero_grad()
        aux = self._entropy_loss(images, prompts)
        aux.backward(retain_graph=True)   # retain — needed for SBCT grad in pse path

        self._save_and_perturb()

        self.base_optimizer.zero_grad()
        pse = self._consistency_loss(images, prompts)
        pse.backward()

        cos = self._cosine()

        self._restore()

        scale = 0.25 * (cos + 1.0) ** 2          # custom_activation
        for group in self.base_optimizer.param_groups:
            group['lr'] = self.init_lr * scale.item()

        self.base_optimizer.step()

        return aux.item(), pse.item(), cos.item()



def train_sam_grata(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    all_dataloader: DataLoader,
):
    """
    Single-epoch TTA loop — GraTa gradient alignment + SBCT + recursive box refinement.

    For every batch:
      1. Apply SBCT (learnable Bezier intensity transform) to the input image.
      2. Recursive box refinement: extract tighter boxes from SAM's first-pass
         mask, re-predict, and keep refined prompts when IoU improves.
      3. GraTa step (entropy + consistency with gradient alignment).
      4. Evaluate (Dice, ASSD, HD95) and save prediction masks.
    """
    raw_model = model.module if hasattr(model, 'module') else model
    trainable = [p for p in raw_model.model.parameters() if p.requires_grad]
    base_opt = optim.Adam(trainable, lr=cfg.opt.learning_rate,
                          weight_decay=cfg.opt.weight_decay)
    grata = GraTaSAM(trainable, base_opt, model, fabric.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_auto = LearnableBezierTransform().to(device)
    optimizer_auto = optim.Adam(transform_auto.parameters(), lr=0.1)

    dice_meter = AverageMeter()
    assd_meter = AverageMeter()
    hd95_meter = AverageMeter()
    ent_meter = AverageMeter()
    consis_meter = AverageMeter()
    cos_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    case_results = []

    for iter, data in enumerate(all_dataloader):
        if getattr(cfg, 'limit', None) and iter >= int(cfg.limit):
            break
        data_time.update(time.time() - end)

        (images_test, bboxes_test, bboxes_test_coarse,
         gt_masks_test, basenames,
         _images_weak, _images_strong,
         _bboxes, _gt_masks) = data

        num_insts = sum(len(m) for m in gt_masks_test)
        if num_insts > cfg.max_nums:
            bboxes_test, gt_masks_test = reduce_instances(
                bboxes_test, gt_masks_test, cfg.max_nums)

        transformed_images = transform_auto(
            images_test[0], basenames[0]).unsqueeze(0)

        prompts_init = get_prompts(cfg, bboxes_test, gt_masks_test)

        raw_model.model.eval()
        with torch.no_grad():
            _, first_masks, first_iou_predictions, _ = model(
                transformed_images, prompts_init)

        batch_size = transformed_images.size(0)
        refined_bboxes_list = []
        for i in range(batch_size):
            mask_2d = (first_masks[i][0] > 0).cpu().numpy().astype(
                np.uint8).squeeze()
            iou_score = first_iou_predictions[i].mean().item()
            non_zero_pts = cv2.findNonZero(mask_2d)

            if iou_score > 0.2 and non_zero_pts is not None:
                x, y, w, h = cv2.boundingRect(non_zero_pts)
                x1, y1, x2, y2 = expand_bbox(
                    x, y, w, h, 1024, 1024, expand_ratio=0.0)
                refined_box = torch.tensor(
                    [[x1, y1, x2, y2]], device=fabric.device,
                    dtype=torch.float32)
                refined_bboxes_list.append(refined_box)
            else:
                refined_bboxes_list.append(
                    bboxes_test[i].to(fabric.device).float())

        prompts_refined = get_prompts(
            cfg, refined_bboxes_list, gt_masks_test)
        with torch.no_grad():
            _, pred_masks2, ious2, _ = model(
                transformed_images, prompts_refined)
            flag = ious2[0].mean().item() > first_iou_predictions[0].mean(
            ).item()

        prompts_refined = prompts_refined if flag else prompts_init

        if flag:
            refined_bboxes_list = []
            for i in range(batch_size):
                mask_2d = (pred_masks2[i][0] > 0).cpu().numpy().astype(
                    np.uint8).squeeze()
                iou_score = ious2[i].mean().item()
                non_zero_pts = cv2.findNonZero(mask_2d)

                if iou_score > 0.2 and non_zero_pts is not None:
                    x, y, w, h = cv2.boundingRect(non_zero_pts)
                    x1, y1, x2, y2 = expand_bbox(
                        x, y, w, h, 1024, 1024, expand_ratio=0.0)
                    refined_box = torch.tensor(
                        [[x1, y1, x2, y2]], device=fabric.device,
                        dtype=torch.float32)
                    refined_bboxes_list.append(refined_box)
                else:
                    refined_bboxes_list.append(
                        bboxes_test[i].to(fabric.device).float())

            prompts_refined = get_prompts(
                cfg, refined_bboxes_list, gt_masks_test)
            with torch.no_grad():
                _, pred_masks3, ious3, _ = model(
                    transformed_images, prompts_refined)
                if ious3[0].mean().item() > first_iou_predictions[0].mean(
                ).item():
                    prompts_refined = get_prompts(
                        cfg, refined_bboxes_list, gt_masks_test)
                else:
                    prompts_refined = prompts_init

        raw_model.model.eval()
        with torch.no_grad():
            _, pred_masks_pre, _, _ = model(
                transformed_images, prompts_refined)

        pre_dice = None
        for pred_mask, gt_mask in zip(pred_masks_pre, gt_masks_test):
            pred_mask_bin = (pred_mask > 0.5).float()
            gt_mask_bin = (gt_mask > 0.5).float()
            if pred_mask_bin.dim() > 2:
                pred_mask_bin = pred_mask_bin.max(dim=0).values
            if gt_mask_bin.dim() > 2:
                gt_mask_bin = gt_mask_bin.max(dim=0).values
            pre_dice, pre_assd, pre_hd95 = calculate_metrics(
                pred_mask_bin, gt_mask_bin)
            break

        if pre_dice is not None and pre_dice == 0:
            fabric.print(f'{basenames[0]} Dice=0, skipping GraTa update.')
            bs = transformed_images.size(0)
            batch_time.update(time.time() - end)
            end = time.time()
            dice_meter.update(0.0, bs)
            assd_meter.update(pre_assd, bs)
            hd95_meter.update(pre_hd95, bs)
            ent_meter.update(0.0, bs)
            consis_meter.update(0.0, bs)
            cos_meter.update(0.0, bs)
            optimizer_auto.step()
            optimizer_auto.zero_grad()
            for basename in basenames:
                case_results.append({
                    "basename": basename, "Dice": 0.0,
                    "ASSD": pre_assd, "HD95": pre_hd95,
                })
                pm_np = (pred_mask_bin.squeeze().cpu().numpy() * 255).astype(
                    np.uint8)
                save_dir = os.path.join(
                    cfg.out_dir,
                    f"{cfg.dataset}-{cfg.prompt}-pred_masks")
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(pm_np).save(os.path.join(save_dir,
                                                         f"{basename}.png"))
            torch.cuda.empty_cache()
            continue

        raw_model.model.train()
        aux, pse, cos = grata.step(transformed_images, prompts_refined)

        optimizer_auto.step()
        optimizer_auto.zero_grad()

        raw_model.model.eval()
        with torch.no_grad():
            _, pred_masks, _, _ = model(transformed_images, prompts_refined)

        for pred_mask, gt_mask, basename in zip(pred_masks, gt_masks_test,
                                                basenames):
            pred_mask = (pred_mask > 0.5).float()
            gt_mask = (gt_mask > 0.5).float()

            if pred_mask.dim() > 2:
                pred_mask = pred_mask.max(dim=0).values
            if gt_mask.dim() > 2:
                gt_mask = gt_mask.max(dim=0).values

            batch_dice, batch_assd, batch_hd95 = calculate_metrics(
                pred_mask, gt_mask)

            num_images = transformed_images.size(0)
            dice_meter.update(batch_dice, num_images)
            assd_meter.update(batch_assd, num_images)
            hd95_meter.update(batch_hd95, num_images)

            pm_save = pred_mask.squeeze()
            if pm_save.dim() > 2:
                pm_save = pm_save.max(dim=0).values
            pm_np = (pm_save.cpu().numpy() * 255).astype(np.uint8)

            save_dir = os.path.join(cfg.out_dir,
                                    f"{cfg.dataset}-{cfg.prompt}-pred_masks")
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(pm_np).save(os.path.join(save_dir,
                                                     f"{basename}.png"))

            case_results.append({
                "basename": basename,
                "Dice": batch_dice,
                "ASSD": batch_assd,
"HD95": batch_hd95,
                "gt_empty": LAST_EVAL["gt_empty"], "pred_empty": LAST_EVAL["pred_empty"],
                })

        bs = transformed_images.size(0)
        ent_meter.update(aux, bs)
        consis_meter.update(pse, bs)
        cos_meter.update(cos, bs)
        batch_time.update(time.time() - end)
        end = time.time()

        fabric.print(
            f'[{iter+1}/{len(all_dataloader)}]'
            f' | Dataset [{cfg.dataset} - {cfg.prompt}]'
            f' | Time [{batch_time.val:.2f}s ({batch_time.avg:.2f}s)]'
            f' | Ent [{ent_meter.val:.4f} ({ent_meter.avg:.4f})]'
            f' | Consis [{consis_meter.val:.4f} ({consis_meter.avg:.4f})]'
            f' | Cos [{cos_meter.val:.3f} ({cos_meter.avg:.3f})]'
            f' | Dice [{dice_meter.val:.4f} ({dice_meter.avg:.4f})]'
        )

        torch.cuda.empty_cache()

    fabric.print(
        f'\n{"="*60}\n'
        f'RESULTS — {cfg.dataset}  ({cfg.prompt})\n'
        f'  Mean Dice : {dice_meter.avg:.4f}\n'
        f'  Mean ASSD : {assd_meter.avg:.4f}\n'
        f'  Mean HD95 : {hd95_meter.avg:.4f}\n'
        f'{"="*60}\n'
    )

    case_results.append({
        "basename": "Average",
        "Dice": dice_meter.avg,
        "ASSD": assd_meter.avg,
        "HD95": hd95_meter.avg,
    })

    if fabric.global_rank == 0:
        csv_path = os.path.join(
            cfg.out_dir,
            f"{cfg.dataset}-{cfg.prompt}-test-results.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(case_results).to_csv(csv_path, index=False)

        ckpt_dir = os.path.join(cfg.out_dir, "save-ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        state = {"model": raw_model}
        ckpt_path = os.path.join(
            ckpt_dir, f"{cfg.dataset}-{cfg.prompt}-last-ckpt.pth")
        torch.save(state, ckpt_path)



def load_config(cfg_path: str, args: argparse.Namespace) -> Box:
    config_module = importlib.import_module(
        cfg_path.replace('.py', '').replace('/', '.'))
    cfg = Box(config_module.config)
    if hasattr(config_module, 'base_config'):
        cfg.merge_update(config_module.base_config)
    if args.dataset:
        cfg.dataset = args.dataset
    if args.prompt:
        cfg.prompt = args.prompt
    if args.gpu_ids:
        cfg.gpu_ids = args.gpu_ids
    cfg.out_dir = "output/grata_tta_add/" + (args.dataset or "default")
    for _key in ('box_csv', 'box_offset', 'seed', 'limit'):
        if getattr(args, _key, None) is not None:
            cfg[_key] = getattr(args, _key)
    if getattr(args, 'out_dir', None):
        cfg.out_dir = args.out_dir
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(cfg: Box) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(
        accelerator="auto",
        devices=num_devices,
        strategy="auto",
        loggers=[TensorBoardLogger(cfg.out_dir,
                                   name=f"{cfg.dataset}-{cfg.prompt}")],
    )
    fabric.launch()
    _seed = getattr(cfg, 'seed', None)
    fabric.seed_everything((_seed if _seed is not None else 1337) + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(cfg.out_dir, "save-ckpt"), exist_ok=True)
        yaml_path = os.path.join(cfg.out_dir, "configs",
                                 f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(cfg.to_dict(), f)

    with fabric.device:
        model = Model(cfg)
        model.setup_pm()
        LoRA_Sam(model.model, 4)

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(
        cfg, model.model.image_encoder.img_size)

    fabric.print(
        f"All: {len(all_data)}  Train: {len(train_data)}  "
        f"Val: {len(val_data)}  Test: {len(test_data)}"
    )

    all_data = fabric._setup_dataloader(all_data)
    model = fabric.setup(model)

    train_sam_grata(cfg, fabric, model, all_data)

    del model, all_data, train_data, val_data, test_data



DEFAULT_DATASETS = [
    "BraTS_PED_t2w_2D",
    "BraTS_PED_t2f_2D",
    "BraTS_SSA_t2w_2D",
    "BraTS_SSA_t2f_2D",
    "ISIC",
]


def run_multiple(cfg: Box, dataset_list):
    """Run TTA on several datasets in sequence, print a final summary."""

    summary_rows = []

    for idx, ds in enumerate(dataset_list, 1):
        print(f"\n{'#'*60}\n"
              f"  Dataset {idx}/{len(dataset_list)} : {ds}\n"
              f"{'#'*60}")
        cfg.dataset = ds
        cfg.out_dir = "output/grata_tta_add/" + ds
        torch.cuda.empty_cache()
        main(cfg)

        csv_path = os.path.join(
            cfg.out_dir, f"{ds}-{cfg.prompt}-test-results.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline='') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if row.get('basename') == 'Average':
                            summary_rows.append((
                                ds,
                                float(row['Dice']),
                                float(row['ASSD']),
                                float(row['HD95']),
                            ))
                            break
                print(f"  -> Dice={summary_rows[-1][1]:.4f}  "
                      f"ASSD={summary_rows[-1][2]:.4f}  "
                      f"HD95={summary_rows[-1][3]:.4f}")
            except Exception as e:
                print(f"  WARNING: Could not read CSV for {ds}: {e}")
        else:
            print(f"  WARNING: Results CSV not found for {ds}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — GraTa TTA (add)")
    print("=" * 70)
    print(f"{'Dataset':<28s} {'Dice':>10s} {'ASSD':>10s} {'HD95':>10s}")
    print("-" * 70)
    for ds, d, a, h in summary_rows:
        print(f"{ds:<28s} {d:>10.4f} {a:>10.4f} {h:>10.4f}")
    print("-" * 70)
    if summary_rows:
        avg_d = np.mean([r[1] for r in summary_rows])
        avg_a = np.mean([r[2] for r in summary_rows])
        avg_h = np.mean([r[3] for r in summary_rows])
        print(f"{'MEAN':<28s} {avg_d:>10.4f} {avg_a:>10.4f} {avg_h:>10.4f}")
    print("=" * 70)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraTa TTA for SAM-based medical image segmentation")

    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config Python file '
                             '(e.g. configs/config_brats.py)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets (overrides --dataset)')
    parser.add_argument('--prompt', type=str, default='box',
                        help='Prompt type (default: box)')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs, comma-separated (default: 0)')
    parser.add_argument('--box_csv', type=str, default=None)
    parser.add_argument('--box_offset', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    set_seed(1337)

    if args.datasets:
        run_multiple(cfg, args.datasets)
    elif args.dataset:
        main(cfg)
    else:
        print("No --dataset/--datasets provided. "
              "Running default five-dataset benchmark.\n"
              f"  Datasets: {DEFAULT_DATASETS}")
        run_multiple(cfg, DEFAULT_DATASETS)
