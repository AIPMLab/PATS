"""TACT baseline (sleeping-weight updates) ported to box-prompted SAM."""
import argparse
import importlib.util
import os
import yaml
import time
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from box import Box
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets import call_load_dataset
from model import Model
from utils.eval_utils import AverageMeter, get_prompts, calculate_metrics, LAST_EVAL
from utils.tools import create_csv, reduce_instances
from PIL import Image
import numpy as np
import pandas as pd


class SaRAAdamW(torch.optim.Optimizer):
    """Masked AdamW (port of TACT/SaRA adamw2.py).

    After each Adam step the update is erased outside the sleeping mask:
        param.mul_(mask); param.add_(ori_param * (1 - mask))
    target_phase=True refreshes the mask (|theta|<tau INTERSECT original).
    """

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, threshold=6e-4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        assert len(self.param_groups) == 1, "single param group expected"
        self.threshold = threshold
        self.inner_iter = 0
        self.param_mask_groups = self.set_mask_param(threshold)
        self.param_mask_groups_adaptive = None

    @torch.no_grad()
    def set_mask_param(self, threshold):
        return [(p.abs() < threshold).bool().requires_grad_(False)
                for p in self.param_groups[0]['params']]

    @torch.no_grad()
    def update_mask(self):
        self.param_mask_groups_adaptive = self.set_mask_param(self.threshold)
        for idx in range(len(self.param_mask_groups_adaptive)):
            self.param_mask_groups_adaptive[idx] = \
                self.param_mask_groups_adaptive[idx] * self.param_mask_groups[idx]

    @torch.no_grad()
    def step(self, closure=None, target_phase=False):
        self.inner_iter += 1
        if target_phase:
            self.update_mask()
            masks = self.param_mask_groups_adaptive
        else:
            masks = self.param_mask_groups
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                bias_c1 = 1 - beta1 ** state['step']
                bias_c2 = 1 - beta2 ** state['step']
                ori = p.data.clone()                       # snapshot pre-update
                p.data.mul_(1 - lr * wd)                   # decoupled wd, full tensor
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-lr / bias_c1)
                m = masks[i].to(p.dtype)
                p.data.mul_(m)
                p.data.add_(ori * (1 - m))
                del ori
        return loss


class FFTMemoryBank:
    """FPDM-style low-frequency amplitude bank (TACT utils/memory_fft.py)."""

    def __init__(self, size=40, fft_size=10):
        self.size = size
        self.fft_size = fft_size
        self.keys = []   # flattened low-freq patches
        self.vals = []

    def get_low_freq(self, x):
        _, _, imgH, imgW = x.size()
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp_src = torch.abs(fft)
        amp_src = torch.fft.fftshift(amp_src)
        ch, cw = imgH // 2, imgW // 2
        h = self.fft_size // 2
        return amp_src[:, :, ch - h: ch + h + 1, cw - h: cw + h + 1]

    def swap(self, x, swap_data):
        _, _, imgH, imgW = x.size()
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)
        ch, cw = imgH // 2, imgW // 2
        h = self.fft_size // 2
        amp_src[:, :, ch - h: ch + h + 1, cw - h: cw + h + 1] = swap_data
        amp_src = torch.fft.ifftshift(amp_src)
        fft_src = torch.complex(real=torch.cos(pha_src) * amp_src,
                                imag=torch.sin(pha_src) * amp_src)
        return torch.fft.ifft2(fft_src, dim=(-2, -1), s=[imgH, imgW]).real

    def push(self, low_freq):
        self.keys.append(low_freq.detach().reshape(-1).cpu().numpy())
        self.vals.append(low_freq.detach().clone())
        if len(self.keys) > self.size:
            self.keys.pop(0)
            self.vals.pop(0)

    def get_neighbours(self, low_freq, k=1):
        q = low_freq.detach().reshape(-1).cpu().numpy()
        sims = []
        for key in self.keys:
            denom = (np.linalg.norm(q) * np.linalg.norm(key)) + 1e-12
            sims.append(float(np.dot(q, key) / denom))
        order = np.argsort(sims)              # ascending (desc=False in TACT)
        idx = order[:k]
        return self.vals[int(idx[0])], float(sims[int(idx[0])])


def sigmoid_entropy_map(logits):
    """TACT's one-sided sigmoid entropy -(p * log(p + 1e-30))."""
    p = torch.sigmoid(logits)
    return -(p * torch.log(p + 1e-30))


def get_entropy_norm(logits):
    ent = sigmoid_entropy_map(logits)
    lo, hi = ent.min(), ent.max()
    if (hi - lo).item() < 1e-12:
        return torch.zeros_like(ent)
    return (ent - lo) * (1.0 / (hi - lo))


def logit2mask_en(logits, threshold=0.5, en_threshold=0.75):
    """Pseudo-label: sigmoid>=thr OR normalized-entropy>=en_threshold."""
    pseudo = (torch.sigmoid(logits.clone()) >= threshold).float()
    ent = get_entropy_norm(logits.clone())
    high = (ent >= en_threshold).float()
    return ((pseudo.int() | high.int())).float()


def logit2mask(logits, threshold=0.5):
    return (torch.sigmoid(logits.clone()) >= threshold).float()


def logit2masked_self(logits, max_seeds=1024):
    """Entropy-guided input mask: boxes around top-entropy seeds (TACT)."""
    ent = get_entropy_norm(logits)
    ent[torch.isnan(ent)] = 0
    b, c, height, width = ent.size()
    masks = torch.zeros_like(ent)
    for bi in range(b):
        for ci in range(c):
            e = ent[bi, ci]
            threshold = 1 - torch.quantile(e.flatten(), 0.1)
            valid = (e > threshold)
            indices = valid.nonzero(as_tuple=False)
            num_high = indices.size(0)
            if num_high == 0:
                continue
            mask_size = max(int(valid.sum().sqrt().item()), 1)
            Num = max(int(num_high * 0.1), 1)
            Num = min(Num, max_seeds)
            if num_high > Num:
                sel = indices[torch.randperm(num_high, device=logits.device)[:Num]]
            else:
                sel = indices
            for idx in sel:
                y_pos, x_pos = int(idx[0].item()), int(idx[1].item())
                y_min, y_max = max(0, y_pos - mask_size // 2), min(height, y_pos + mask_size // 2 + 1)
                x_min, x_max = max(0, x_pos - mask_size // 2), min(width, x_pos + mask_size // 2 + 1)
                masks[bi, ci, y_min:y_max, x_min:x_max] = 1
    return masks


def augment(x, factor):
    if factor == 0:
        return x.flip(-1)
    elif factor == 1:
        return x.flip(-2)
    elif factor == 2:
        return x.flip(-1).transpose(-2, -1)
    elif factor == 3:
        return x.flip(-1).flip(-2)
    elif factor == 4:
        return x.transpose(-2, -1).flip(-1)
    elif factor == 5:
        return x
    raise ValueError(factor)


def inverse_augment(output, factor):
    inverse_map = {0: 0, 1: 1, 2: 4, 3: 3, 4: 2, 5: 5}
    return augment(output, inverse_map[factor])


def dice_loss_repo(pred, label):
    smooth = 1.
    m1 = pred.contiguous().view(-1)
    m2 = label.contiguous().view(-1)
    inter = (m1 * m2).sum()
    return 1 - ((2. * inter + smooth) / (m1.sum() + m2.sum() + smooth))


def soft_ce_distill(x_logits, x_ema_logits):
    """TACT softmax_entropy: (-p_student * log(p_teacher + 1e-8)).sum(1)."""
    x_prob = x_logits.sigmoid()
    x_ema_prob = x_ema_logits.sigmoid()
    return (-x_prob * torch.log(x_ema_prob + 1e-8)).sum(1)


class PixelContrast:
    """Compact port of TACT's pixel contrast (temp 0.5, queue 500/class)."""

    def __init__(self, dim=256, n_classes=2, temperature=0.5,
                 queue_size=500, max_views=5):
        self.temperature = temperature
        self.n_classes = n_classes
        self.max_views = max_views
        self.dim = dim
        self.queue = {c: None for c in range(n_classes)}
        self.queue_size = queue_size

    @torch.no_grad()
    def _enqueue(self, feats, labels):
        for c in range(self.n_classes):
            idx = (labels == c)
            if idx.sum() == 0:
                continue
            fc = feats[idx]
            if fc.shape[0] > self.queue_size:
                fc = fc[torch.randperm(fc.shape[0])[:self.queue_size]]
            if self.queue[c] is None:
                self.queue[c] = fc.clone()
            else:
                self.queue[c] = torch.cat([self.queue[c], fc], 0)[-self.queue_size:]

    def __call__(self, feats, labels_low, predict, en_map, labels_hard):
        """feats: (1,dim,h,w) normalized outside; labels_low/hard & predict &
        en_map: (1,1,H,W) binary/normalized maps at image resolution
        (multi-box channels must be union-collapsed by the caller)."""
        h, w = feats.shape[-2:]
        for t in (labels_low, predict, en_map, labels_hard):
            assert t.size(1) == 1, "PCL maps must be single-channel"
        lab = F.interpolate(labels_low, (h, w), mode='nearest')[0, 0].long()
        pred = F.interpolate(predict, (h, w), mode='nearest')[0, 0].long()
        enm = F.interpolate(en_map, (h, w), mode='nearest')[0, 0]
        f = feats[0].permute(1, 2, 0).reshape(-1, self.dim)   # (h*w, dim)
        sel_feats, sel_labels = [], []
        for c in range(self.n_classes):
            hard = ((pred == c) & (lab != c)).nonzero(as_tuple=False)
            easy = ((pred == c) & (lab == c)).nonzero(as_tuple=False)
            n_view = self.max_views
            if hard.shape[0] >= n_view:
                order = torch.argsort(enm[hard[:, 0], hard[:, 1]], descending=True)
                chosen = hard[order[:n_view]]
            elif hard.shape[0] + easy.shape[0] >= n_view:
                chosen = torch.cat([hard, easy[:n_view - hard.shape[0]]], 0)
            elif hard.shape[0] + easy.shape[0] > 0:
                chosen = torch.cat([hard, easy], 0)
            else:
                continue
            sel_feats.append(f[chosen[:, 0] * w + chosen[:, 1]])
            sel_labels.append(torch.full((chosen.shape[0],), c, dtype=torch.long,
                                         device=feats.device))
        if not sel_feats:
            self._enqueue(f.detach(), F.interpolate(labels_hard, (h, w),
                                                    mode='nearest')[0, 0].long().reshape(-1))
            return torch.tensor(0., device=feats.device)
        anchors = torch.cat(sel_feats, 0)
        a_labels = torch.cat(sel_labels, 0)
        queue_feats, queue_labels = [], []
        for c in range(self.n_classes):
            if self.queue[c] is not None and self.queue[c].shape[0] > 0:
                queue_feats.append(self.queue[c])
                queue_labels.append(torch.full((self.queue[c].shape[0],), c,
                                               dtype=torch.long, device=feats.device))
        self._enqueue(f.detach(), F.interpolate(labels_hard, (h, w),
                                                mode='nearest')[0, 0].long().reshape(-1))
        if not queue_feats:
            return torch.tensor(0., device=feats.device)
        q_f = torch.cat(queue_feats, 0)
        q_l = torch.cat(queue_labels, 0)
        logits = anchors @ q_f.t() / self.temperature
        return F.cross_entropy(logits, a_labels)


def train_sam(
    cfg: Box,
    fabric,
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
            x = images_test

            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)
            prompts = get_prompts(cfg, bboxes, gt_masks)          # clean boxes

            model.model.load_state_dict(target_sam.state_dict())

            model.eval()
            prompts_test = get_prompts(cfg, bboxes_test, gt_masks_test)
            with torch.no_grad():
                _, pred_masks_eval, _, _ = model(x, prompts_test)
            for pred_mask, gt_mask, basename in zip(pred_masks_eval, gt_masks_test, basenames):
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
                torch.cuda.empty_cache()
                batch_time.update(time.time() - end)
                end = time.time()
                case_results.append({"basename": basename, "Dice": batch_dice, "ASSD": batch_assd, "HD95": batch_hd95, "gt_empty": LAST_EVAL["gt_empty"], "pred_empty": LAST_EVAL["pred_empty"]})
                fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")
                continue

            optimizer.zero_grad()

            with torch.no_grad():
                t_embeds, t_preds, _, _ = teacher_model(x, prompts)
                tl = t_preds[0] if t_preds[0].dim() == 3 else t_preds[0][0]  # (n,H,W)
                fea_t = t_embeds[0]
            with torch.no_grad():
                pseudo_t = logit2mask_en(tl, thr_t_mask, thr_pcl_en)        # (n,H,W)
                mask_en = logit2masked_self(tl.unsqueeze(0))                 # (1,n,H,W)
                if mask_en.size(1) > 1:
                    mask_en = mask_en.max(1, keepdim=True)[0]                # union

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
            sl = s_preds[0] if s_preds[0].dim() == 3 else s_preds[0][0]      # (n,H,W)
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

            model.eval()
            with torch.no_grad():
                ens_list = []
                for factor in [5, 0, 1, 2, 3, 4]:
                    xa = augment(x, factor)
                    _, preds_a, _, _ = model(xa, prompts)
                    la = preds_a[0] if preds_a[0].dim() == 3 else preds_a[0][0]
                    ens_list.append(inverse_augment(la, factor).unsqueeze(0))
                t2 = torch.cat(ens_list, 0).mean(0, keepdim=True)            # (1,n,H,W)
            model.train()

            s2_embeds, s2_preds, _, _ = model(x, prompts)
            s2 = (s2_preds[0] if s2_preds[0].dim() == 3
                  else s2_preds[0][0]).unsqueeze(0)                          # (1,n,H,W)
            distill = soft_ce_distill(s2, t2.detach()).mean()
            en_map_t = get_entropy_norm(t2.detach())
            pl_s = (torch.sigmoid(t2.detach()) >= thr_pcl_pre).float()
            hard_pseudo = (pl_s.int() | (en_map_t >= thr_pcl_en).int()).float()
            low_pseudo = (pl_s.int() & (en_map_t < thr_pcl_en).int()).float()
            predict_bin = (torch.sigmoid(s2.detach()) >= thr_pcl_pre).float()

            def _u(t):
                return t.max(1, keepdim=True)[0] if t.size(1) > 1 else t

            pcl_feat = F.normalize(s2_embeds[0], p=2, dim=0).unsqueeze(0)    # (1,256,64,64)
            contrast = contrast_criterion(pcl_feat, _u(low_pseudo), _u(predict_bin),
                                          _u(en_map_t), _u(hard_pseudo))
            loss2 = 0.1 * contrast + 0.1 * distill
            fabric.backward(loss2)
            optimizer.step(target_phase=True)
            optimizer.zero_grad()
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
        fabric.print(f"[TACT] sleeping weights {n_sleep}/{n_total} "
                     f"({100.0 * n_sleep / max(n_total, 1):.3f}%) "
                     f"tau={thr} lr={lr}")

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(cfg, model.model.image_encoder.img_size)
    contrast_criterion = PixelContrast(dim=256, n_classes=2, temperature=0.5,
                                       queue_size=500, max_views=5)
    fabric.print(f"All Data: {len(all_data) * cfg.batch_size}")
    all_data = fabric._setup_dataloader(all_data)
    model = fabric.setup(model)

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
    cfg.out_dir = "output/tact_tta/" + args.dataset
    for _key in ('box_csv', 'box_offset', 'seed', 'limit', 'tact_lr',
                 'sparse_thr', 'bank_size'):
        if getattr(args, _key, None) is not None:
            cfg[_key] = getattr(args, _key)
    if getattr(args, 'out_dir', None):
        cfg.out_dir = args.out_dir
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TACT (CVPR 2026F) on SAM")
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--gpu_ids', type=str)
    parser.add_argument('--box_csv', type=str, default=None)
    parser.add_argument('--box_offset', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--tact_lr', type=float, default=None,
                        help='SaRA AdamW lr (default 1e-5, ViT setting)')
    parser.add_argument('--sparse_thr', type=float, default=None,
                        help='sleeping-weight threshold tau (default 6e-4)')
    parser.add_argument('--bank_size', type=int, default=None,
                        help='FFT memory bank size (default 40)')
    args = parser.parse_args()
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
