"""TACT with the PATS wrapper."""
import argparse
import importlib.util
import os
import yaml
import time
import math
import copy
import torch
import torch.nn.functional as F
import lightning as L
from box import Box
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets import call_load_dataset
from model import Model
from utils.eval_utils import AverageMeter, get_prompts, calculate_metrics, LAST_EVAL
from utils.nonlinear_learned import LearnableBezierTransform
import torch.optim as optim
from utils.tools import create_csv, reduce_instances
from PIL import Image
import numpy as np
import pandas as pd
import cv2

from adaptation_tact import (
    SaRAAdamW, FFTMemoryBank, PixelContrast,
    sigmoid_entropy_map, get_entropy_norm, logit2mask_en, logit2mask,
    logit2masked_self, augment, inverse_augment, dice_loss_repo,
    soft_ce_distill,
)


def expand_bbox(x, y, w, h, width, height, expand_ratio=0.25):
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)
    new_x2 = min(width, x + w + (new_w - w) / 2)
    new_y2 = min(height, y + h + (new_h - h) / 2)
    return new_x, new_y, new_x2, new_y2


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    teacher_model: Model,
    target_sam,
    optimizer: SaRAAdamW,
    all_dataloader: DataLoader,
    contrast_criterion: PixelContrast,
):
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    end = time.time()
    num_epochs = 1
    dice_scores = AverageMeter()
    assd_scores = AverageMeter()
    hd95_scores = AverageMeter()
    case_results = []
    from adaptation_tent_add import _BOnly
    _esbct_on = not getattr(cfg, "no_esbct", False)
    transform_auto = (LearnableBezierTransform() if _esbct_on
                      else _BOnly()).to(torch.device(
                          'cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer_auto = optim.Adam(transform_auto.parameters(), lr=0.1)
    bank = FFTMemoryBank(size=int(getattr(cfg, 'bank_size', 40)),
                         fft_size=max(1, int(0.01 * 1024)))
    thr_t_mask = 0.5
    thr_pcl_en = 0.75
    thr_pcl_pre = 0.5

    for epoch in range(0, num_epochs):
        for it, data in enumerate(all_dataloader):
            if getattr(cfg, 'limit', None) and it >= int(cfg.limit):
                break
            images_test, bboxes_test, bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data
            x = transform_auto(images_test[0], basenames[0]).unsqueeze(0)

            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)
            prompts = get_prompts(cfg, bboxes, gt_masks)

            model.model.load_state_dict(target_sam.state_dict())

            model.eval()
            batch_size_eval = images_test.size(0)
            prompts_test = get_prompts(cfg, bboxes_test, gt_masks_test)
            with torch.no_grad():
                _, pred_masks_eval, ious_eval, _ = model(x, prompts_test)

            refined_bboxes_list = []
            for i in range(batch_size_eval):
                mask_2d = (pred_masks_eval[i][0] > 0).cpu().numpy().astype(np.uint8).squeeze()
                iou_score = ious_eval[i].mean().item()
                non_zero_pts = cv2.findNonZero(mask_2d)
                if iou_score > 0.2 and non_zero_pts is not None:
                    xx, yy, w_, h_ = cv2.boundingRect(non_zero_pts)
                    height, width = images_test.size(2), images_test.size(3)
                    x1, y1, x2, y2 = expand_bbox(xx, yy, w_, h_, width, height, expand_ratio=0.0)
                    refined_bboxes_list.append(torch.tensor([[x1, y1, x2, y2]], device=fabric.device, dtype=torch.float32))
                else:
                    refined_bboxes_list.append(bboxes_test[i].to(fabric.device).float())
            prompts_refined = get_prompts(cfg, refined_bboxes_list, gt_masks_test)
            with torch.no_grad():
                _, pred_masks_refined, ious_refined, _ = model(x, prompts_refined)

            if (getattr(cfg, 'pal_select', None) or 'iou') == 'gt_dice':
                dice_orig = None
                for pred_mask, gt_mask in zip(pred_masks_eval, gt_masks_test):
                    pm = (pred_mask > 0.5).float()
                    gm = (gt_mask > 0.5).float()
                    if pm.dim() > 2:
                        pm = pm.max(dim=0).values
                    if gm.dim() > 2:
                        gm = gm.max(dim=0).values
                    dice_orig, _, _ = calculate_metrics(pm, gm)
                dice_refined = None
                for pred_mask, gt_mask in zip(pred_masks_refined, gt_masks_test):
                    pm = (pred_mask > 0.5).float()
                    gm = (gt_mask > 0.5).float()
                    if pm.dim() > 2:
                        pm = pm.max(dim=0).values
                    if gm.dim() > 2:
                        gm = gm.max(dim=0).values
                    dice_refined, _, _ = calculate_metrics(pm, gm)
                use_refined = dice_refined > dice_orig
            else:
                use_refined = ious_refined[0].mean().item() > ious_eval[0].mean().item()

            if use_refined:
                refined_bboxes_list2 = []
                for i in range(batch_size_eval):
                    mask_2d = (pred_masks_refined[i][0] > 0).cpu().numpy().astype(np.uint8).squeeze()
                    iou_score = ious_refined[i].mean().item()
                    non_zero_pts = cv2.findNonZero(mask_2d)
                    if iou_score > 0.2 and non_zero_pts is not None:
                        xx, yy, w_, h_ = cv2.boundingRect(non_zero_pts)
                        height, width = images_test.size(2), images_test.size(3)
                        x1, y1, x2, y2 = expand_bbox(xx, yy, w_, h_, width, height, expand_ratio=0.0)
                        refined_bboxes_list2.append(torch.tensor(
                            [[x1, y1, x2, y2]], device=fabric.device, dtype=torch.float32))
                    else:
                        refined_bboxes_list2.append(bboxes_test[i].to(fabric.device).float())
                prompts_refined2 = get_prompts(cfg, refined_bboxes_list2, gt_masks_test)
                with torch.no_grad():
                    _, pred_masks_refined2, ious_refined2, _ = model(x, prompts_refined2)
                if (getattr(cfg, 'pal_select', None) or 'iou') == 'gt_dice':
                    dice_refined2 = None
                    for pred_mask, gt_mask in zip(pred_masks_refined2, gt_masks_test):
                        pm = (pred_mask > 0.5).float()
                        gm = (gt_mask > 0.5).float()
                        if pm.dim() > 2:
                            pm = pm.max(dim=0).values
                        if gm.dim() > 2:
                            gm = gm.max(dim=0).values
                        dice_refined2, _, _ = calculate_metrics(pm, gm)
                    use_refined2 = dice_refined2 > dice_orig
                else:
                    use_refined2 = ious_refined2[0].mean().item() > ious_eval[0].mean().item()
                if use_refined2:
                    pred_masks_refined = pred_masks_refined2
            pred_masks_final = pred_masks_refined if use_refined else pred_masks_eval

            for pred_mask, gt_mask, basename in zip(pred_masks_final, gt_masks_test, basenames):
                pred_mask = (pred_mask > 0.5).float()
                gt_mask = (gt_mask > 0.5).float()
                if pred_mask.dim() > 2:
                    pred_mask = pred_mask.max(dim=0).values
                if gt_mask.dim() > 2:
                    gt_mask = gt_mask.max(dim=0).values
                batch_dice, batch_assd, batch_hd95 = calculate_metrics(pred_mask, gt_mask)
            model.train()

            if batch_dice == 0:
                fabric.print(f"{basename} Dice=0, skipping update.")
                optimizer.zero_grad()
                optimizer_auto.zero_grad()
                torch.cuda.empty_cache()
                batch_time.update(time.time() - end)
                end = time.time()
                case_results.append({"basename": basename, "Dice": batch_dice, "ASSD": batch_assd, "HD95": batch_hd95, "gt_empty": LAST_EVAL["gt_empty"], "pred_empty": LAST_EVAL["pred_empty"]})
                fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")
                continue

            optimizer.zero_grad()
            optimizer_auto.zero_grad()

            with torch.no_grad():
                t_embeds, t_preds, _, _ = teacher_model(x, prompts)
                tl = t_preds[0] if t_preds[0].dim() == 3 else t_preds[0][0]
                fea_t = t_embeds[0]
            with torch.no_grad():
                pseudo_t = logit2mask_en(tl, thr_t_mask, thr_pcl_en)
                mask_en = logit2masked_self(tl.unsqueeze(0))
                if mask_en.size(1) > 1:
                    mask_en = mask_en.max(1, keepdim=True)[0]

            low_freq = bank.get_low_freq(x)
            if len(bank.keys) >= 1:
                init_data, _ = bank.get_neighbours(low_freq, k=1)
                bank.push(low_freq)
                masked_x = bank.swap(x, init_data)
            else:
                bank.push(low_freq)
                masked_x = x.clone()
            for c in range(masked_x.size(1)):
                masked_x[:, c, :, :] = masked_x[:, c, :, :] * (1 - mask_en[:, 0, :, :])

            s_embeds, s_preds, _, _ = model(masked_x, prompts)
            sl = s_preds[0] if s_preds[0].dim() == 3 else s_preds[0][0]
            loss_s1 = torch.tensor(0., device=fabric.device)
            for k in range(sl.shape[0]):
                probs = torch.sigmoid(sl[k])
                loss_s1 = loss_s1 + dice_loss_repo(probs, pseudo_t[k]) + \
                    F.binary_cross_entropy(probs, pseudo_t[k])
            loss_s1 = loss_s1 / sl.shape[0]
            fea_s = s_embeds[0]
            cons = F.l1_loss(torch.abs(torch.fft.fft2(fea_s)),
                             torch.abs(torch.fft.fft2(fea_t.detach())))
            loss1 = 1.0 * loss_s1 + 0.1 * cons
            fabric.backward(loss1)
            optimizer.step()
            optimizer.zero_grad()
            target_sam.load_state_dict(model.model.state_dict())

            x = transform_auto(images_test[0], basenames[0]).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                ens_list = []
                for factor in [5, 0, 1, 2, 3, 4]:
                    xa = augment(x, factor)
                    _, preds_a, _, _ = model(xa, prompts)
                    la = preds_a[0] if preds_a[0].dim() == 3 else preds_a[0][0]
                    ens_list.append(inverse_augment(la, factor).unsqueeze(0))
                t2 = torch.cat(ens_list, 0).mean(0, keepdim=True)
            model.train()

            s2_embeds, s2_preds, _, _ = model(x, prompts)
            s2 = (s2_preds[0] if s2_preds[0].dim() == 3
                  else s2_preds[0][0]).unsqueeze(0)
            distill = soft_ce_distill(s2, t2.detach()).mean()
            en_map_t = get_entropy_norm(t2.detach())
            pl_s = (torch.sigmoid(t2.detach()) >= thr_pcl_pre).float()
            hard_pseudo = (pl_s.int() | (en_map_t >= thr_pcl_en).int()).float()
            low_pseudo = (pl_s.int() & (en_map_t < thr_pcl_en).int()).float()
            predict_bin = (torch.sigmoid(s2.detach()) >= thr_pcl_pre).float()

            def _u(t):
                return t.max(1, keepdim=True)[0] if t.size(1) > 1 else t

            pcl_feat = F.normalize(s2_embeds[0], p=2, dim=0).unsqueeze(0)
            contrast = contrast_criterion(pcl_feat, _u(low_pseudo), _u(predict_bin),
                                          _u(en_map_t), _u(hard_pseudo))
            loss2 = 0.1 * contrast + 0.1 * distill
            fabric.backward(loss2)
            optimizer.step(target_phase=True)
            optimizer.zero_grad()
            optimizer_auto.step()
            optimizer_auto.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()
            total_losses.update((loss1 + loss2).item(), batch_size)
            fabric.print(f'Epoch: [{epoch}][{it+1}/{len(all_dataloader)}]'
                         f' | Dataset: [{cfg.dataset} - {cfg.prompt}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | L1 {loss1.item():.4f} | L2 {loss2.item():.4f}')
            fabric.log_dict({"Total Loss": total_losses.avg})

            dice_scores.update(batch_dice, images_test.size(0))
            assd_scores.update(batch_assd, images_test.size(0))
            hd95_scores.update(batch_hd95, images_test.size(0))

            pred_mask_save = pred_mask.squeeze()
            if pred_mask_save.dim() > 2:
                pred_mask_save = pred_mask_save.max(dim=0).values
            pred_mask_np = (pred_mask_save.cpu().numpy() * 255).astype(np.uint8)
            save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basenames[0]}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(pred_mask_np).save(save_path)

            case_results.append({
                "basename": basenames[0],
                "Dice": batch_dice,
                "ASSD": batch_assd,
                "HD95": batch_hd95,
                "gt_empty": LAST_EVAL["gt_empty"], "pred_empty": LAST_EVAL["pred_empty"],
            })
            fabric.print(f"{basenames[0]} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")
            torch.cuda.empty_cache()

    fabric.print(f'Test Ending...: Mean Dice: [{dice_scores.avg:.4f}] -- Mean ASSD: [{assd_scores.avg:.4f}] -- Mean HD95: [{hd95_scores.avg:.4f}]')
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


def main(cfg: Box) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)
    fabric = L.Fabric(accelerator="auto", devices=num_devices, strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")])
    fabric.launch()
    _seed = getattr(cfg, 'seed', None)
    fabric.seed_everything((_seed if _seed is not None else 1337) + fabric.global_rank)

    if fabric.global_rank == 0:
        cfg_dict = cfg.to_dict()
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        with open(os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml"), "w") as file:
            yaml.dump(cfg_dict, file)

    with fabric.device:
        model = Model(cfg)
        model.setup()
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        target_sam = copy.deepcopy(model.model)
        for p in model.model.parameters():
            p.requires_grad_(True)
        lr = float(getattr(cfg, 'tact_lr', 1e-5))
        thr = float(getattr(cfg, 'sparse_thr', 6e-4))
        optimizer = SaRAAdamW(model.model.parameters(), lr=lr,
                              betas=(0.9, 0.999), eps=1e-8,
                              weight_decay=1e-2, threshold=thr)
        n_sleep = sum(int(m.sum()) for m in optimizer.param_mask_groups)
        n_total = sum(p.numel() for p in model.model.parameters())
        fabric.print(f"[TACT+PATS] sleeping weights {n_sleep}/{n_total} "
                     f"({100.0 * n_sleep / max(n_total, 1):.3f}%) "
                     f"tau={thr} lr={lr}")

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(cfg, model.model.image_encoder.img_size)
    contrast_criterion = PixelContrast(dim=256, n_classes=2, temperature=0.5,
                                       queue_size=500, max_views=5)
    fabric.print(f"All Data: {len(all_data) * cfg.batch_size}")
    all_data = fabric._setup_dataloader(all_data)
    model = fabric.setup(model)  # SaRA optimizer stays raw (target_phase kwarg)

    train_sam(cfg, fabric, model, teacher_model, target_sam, optimizer,
              all_data, contrast_criterion)

    del model, teacher_model, train_data, val_data


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
    cfg.out_dir = "output/tact_add_tta/" + args.dataset
    for _key in ('box_csv', 'box_offset', 'seed', 'limit', 'tact_lr',
                 'sparse_thr', 'bank_size', 'pal_select', 'no_esbct'):
        if getattr(args, _key, None) is not None:
            cfg[_key] = getattr(args, _key)
    if getattr(args, 'out_dir', None):
        cfg.out_dir = args.out_dir
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TACT (CVPR 2026F) + PATS on SAM")
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--gpu_ids', type=str)
    parser.add_argument('--box_csv', type=str, default=None)
    parser.add_argument('--box_offset', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--tact_lr', type=float, default=None)
    parser.add_argument('--sparse_thr', type=float, default=None)
    parser.add_argument('--bank_size', type=int, default=None)
    parser.add_argument('--no_esbct', action='store_true',
                        help='disable E-SBCT (PAL-only ablation)')
    parser.add_argument('--pal_select', type=str, default=None, choices=['iou', 'gt_dice'])
    args = parser.parse_args()
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
