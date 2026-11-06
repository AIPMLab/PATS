"""TENT with the PATS wrapper."""
import argparse
import importlib.util
import os
import yaml
import time
import torch
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset

from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import AverageMeter, calc_iou, validate, get_prompts, validate_med, test_med, calculate_metrics, LAST_EVAL
from utils.nonlinear_learned import LearnableBezierTransform
import torch.optim as optim
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
from PIL import Image
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw
import numpy as np

from PIL import Image, ImageDraw
import numpy as np

from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2


def expand_bbox(x, y, w, h, width, height, expand_ratio=0.25):
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)

    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)

    new_x2 = min(width, x + w + (new_w - w) / 2)
    new_y2 = min(height, y + h + (new_h - h) / 2)

    return new_x, new_y, new_x2, new_y2


def visualize_and_save_points_on_image(image, gt, po_points, na_points, filename):
    image = image * 255
    image_np = image.permute(1, 2, 0).cpu().numpy()
    gt_np = gt[0].cpu().numpy()
    gt_np = gt_np[0]
    gt_np = np.uint8(gt_np * 255)

    gt_pil = Image.fromarray(gt_np)
    gt_pil = gt_pil.resize((image_np.shape[1], image_np.shape[0]))
    gt_np_resized = np.array(gt_pil)

    image_pil = Image.fromarray(np.uint8(image_np))
    draw = ImageDraw.Draw(image_pil)

    point_size = 5

    po_points = po_points[0]
    na_points = na_points[0]

    for point in po_points:
        x, y = int(point[1]), int(point[0])
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill='black')

    for point in na_points:
        x, y = int(point[1]), int(point[0])
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill='blue')

    gt_pil = Image.fromarray(gt_np_resized)
    combined_image = Image.new('RGB', (image_pil.width, image_pil.height + gt_pil.height))
    combined_image.paste(image_pil, (0, 0))
    combined_image.paste(gt_pil.convert('RGB'), (0, image_pil.height))

    combined_image.save(filename)


class _BOnly(torch.nn.Module):
    """PAL-only ablation: three independent intensity Bezier curves."""

    def __init__(self):
        super().__init__()
        self.cp = torch.nn.ParameterList([\
            torch.nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0])),\
            torch.nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0])),\
            torch.nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))])

    def forward(self, x, base_name=None):
        outs = []
        for i, cp in enumerate(self.cp):
            c = torch.sigmoid(cp)
            t = x[i:i + 1] if x.shape[0] >= 3 else x
            omt = 1. - t
            a, b, c0, d = c[0], c[1], c[2], c[3]
            outs.append(torch.clamp(omt**3 * a + 3.0 * t * omt**2 * b
                            + 3.0 * t**2 * omt * c0 + t**3 * d, 0.0, 1.0))
        return torch.cat(outs, dim=0)


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    anchor_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    all_dataloader: DataLoader,
    num_iters: int,
):
    """The SAM training loop with closed-loop box refinement for final result only."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    focal_losses = AverageMeter()
    dice_losses = AverageMeter()
    iou_losses = AverageMeter()
    anchor_losses = AverageMeter()
    contra_losses = AverageMeter()
    total_losses = AverageMeter()
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    contra_loss = ContraLoss()
    end = time.time()
    max_dice = 0.
    num_epochs = 1  ## number of epochs, TTA was set to 1
    dice_scores = AverageMeter()
    assd_scores = AverageMeter()
    hd95_scores = AverageMeter()
    case_results = []
    _esbct_on = not getattr(cfg, "no_esbct", False)
    transform_auto = (LearnableBezierTransform() if _esbct_on
                      else _BOnly()).to(torch.device(
                          'cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer_auto = optim.Adam(transform_auto.parameters(), lr=0.1)

    for epoch in range(0, num_epochs):
        for iter, data in enumerate(all_dataloader):
            if getattr(cfg, 'limit', None) and iter >= int(cfg.limit):
                break
            data_time.update(time.time() - end)
            images_test, bboxes_test, bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data
            transformed_images = transform_auto(\
                images_test[0], basenames[0]).unsqueeze(0)

            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                print(num_insts)
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            prompts = get_prompts(cfg, bboxes, gt_masks)
            soft_image_embeds, pred_masks, soft_iou_predictions, soft_res_masks = model(transformed_images, prompts)

            loss_total = torch.tensor(0., device=fabric.device)

            for pred_mask in pred_masks:
                pred_mask = F.sigmoid(pred_mask)
                pred_mask = torch.clamp(pred_mask, min=0, max=1)

                for mask in pred_mask:
                    mask = mask.view(-1)    # [256 * 256]
                    loss = - mask.softmax(0) * torch.log(mask.softmax(0)) - (1 - mask.softmax(0)) * torch.log(1 - mask.softmax(0))
                    loss_total += loss[~torch.isnan(loss)].sum()
            loss_total = loss_total/100.0

            model.eval()
            num_images = images_test.size(0)
            prompts_test = get_prompts(cfg, bboxes_test, gt_masks_test)
            with torch.no_grad():
                _, pred_masks_eval, ious_eval, _ = model(transformed_images, prompts_test)

            refined_bboxes_list = []
            for i in range(batch_size):
                mask_2d = (pred_masks_eval[i][0] > 0).cpu().numpy().astype(np.uint8).squeeze()
                iou_score = ious_eval[i].mean().item()
                non_zero_pts = cv2.findNonZero(mask_2d)

                if iou_score > 0.2 and non_zero_pts is not None:
                    x, y, w, h = cv2.boundingRect(non_zero_pts)
                    height, width = images_test.size(2), images_test.size(3)
                    x1, y1, x2, y2 = expand_bbox(x, y, w, h, width, height, expand_ratio=0.0)
                    refined_box = torch.tensor([[x1, y1, x2, y2]], device=fabric.device, dtype=torch.float32)
                    refined_bboxes_list.append(refined_box)
                else:
                    refined_bboxes_list.append(bboxes_test[i].to(fabric.device).float())

            prompts_refined = get_prompts(cfg, refined_bboxes_list, gt_masks_test)
            with torch.no_grad():
                _, pred_masks_refined, ious_refined, _ = model(transformed_images, prompts_refined)

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
                for i in range(batch_size):
                    mask_2d = (pred_masks_refined[i][0] > 0).cpu().numpy().astype(np.uint8).squeeze()
                    iou_score = ious_refined[i].mean().item()
                    non_zero_pts = cv2.findNonZero(mask_2d)
                    if iou_score > 0.2 and non_zero_pts is not None:
                        x, y, w, h = cv2.boundingRect(non_zero_pts)
                        height, width = images_test.size(2), images_test.size(3)
                        x1, y1, x2, y2 = expand_bbox(x, y, w, h, width, height, expand_ratio=0.0)
                        refined_bboxes_list2.append(torch.tensor(
                            [[x1, y1, x2, y2]], device=fabric.device, dtype=torch.float32))
                    else:
                        refined_bboxes_list2.append(bboxes_test[i].to(fabric.device).float())
                prompts_refined2 = get_prompts(cfg, refined_bboxes_list2, gt_masks_test)
                with torch.no_grad():
                    _, pred_masks_refined2, ious_refined2, _ = model(transformed_images, prompts_refined2)
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
                torch.cuda.empty_cache()
                batch_time.update(time.time() - end)
                end = time.time()
                case_results.append({"basename": basename, "Dice": batch_dice, "ASSD": batch_assd, "HD95": batch_hd95, "gt_empty": LAST_EVAL["gt_empty"], "pred_empty": LAST_EVAL["pred_empty"]})
                fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")
                continue

            fabric.backward(loss_total)

            optimizer.step()
            optimizer_auto.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_auto.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()


            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(all_dataloader)}]'
                         f' | Dataset: [{cfg.dataset} - {cfg.prompt}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            loss_logger = {"Total Loss": total_losses.avg}
            fabric.log_dict(loss_logger)
            torch.cuda.empty_cache()
            dice_scores.update(batch_dice, num_images)
            assd_scores.update(batch_assd, num_images)
            hd95_scores.update(batch_hd95, num_images)

            pred_mask_save = pred_mask.squeeze()
            if pred_mask_save.dim() > 2:
                pred_mask_save = pred_mask_save.max(dim=0).values
            pred_mask_np = (pred_mask_save.cpu().numpy() * 255).astype(np.uint8)

            save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basename}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            Image.fromarray(pred_mask_np).save(save_path)

            case_results.append({
                "basename": basename,
                "Dice": batch_dice,
                "ASSD": batch_assd,
"HD95": batch_hd95,
                "gt_empty": LAST_EVAL["gt_empty"], "pred_empty": LAST_EVAL["pred_empty"],
                })


            fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")
            del pred_masks_eval
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

    state = {"model": model, "optimizer": optimizer}
    fabric.save(os.path.join(cfg.out_dir, "save-ckpt", f"{cfg.dataset}-{cfg.prompt}-last-ckpt.pth"), state)




def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def multi_main(cfg):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")])
    fabric.launch()
    _seed = getattr(cfg, 'seed', None)
    fabric.seed_everything((_seed if _seed is not None else 1337) + fabric.global_rank)

    if fabric.global_rank == 0:
        cfg_dict = cfg.to_dict()
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        cfg_dict_path = os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(cfg_dict_path, "w") as file:
            yaml.dump(cfg_dict, file)

        os.makedirs(os.path.join(cfg.out_dir, "save-ckpt"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()
        anchor_model = copy_model(model)
        LoRA_Sam(model.model, 4)

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model.model)

    fabric.print(f"All Data: {len(all_data) * cfg.batch_size}; Train Data: {len(train_data) * cfg.batch_size}; Val Data: {len(val_data) * cfg.val_batchsize}; Val Data: {len(test_data) * 1}")
    num_iters = len(train_data) * cfg.batch_size
    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])
    all_data = fabric._setup_dataloader(all_data)
    model, optimizer = fabric.setup(model, optimizer)


    train_sam(cfg, fabric, model, anchor_model, optimizer, scheduler, all_data, num_iters)

    del model, anchor_model, train_data, val_data


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
    cfg.out_dir = "output/tent_add_tta/" + args.dataset
    for _key in ('box_csv', 'box_offset', 'seed', 'limit', 'pal_select', 'no_esbct'):
        if getattr(args, _key, None) is not None:
            cfg[_key] = getattr(args, _key)
    if getattr(args, 'out_dir', None):
        cfg.out_dir = args.out_dir

    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg', type=str, required=True, help='Path to the configuration Python file (e.g., configs/config.py)')
    parser.add_argument('--dataset', type=str, help='Dataset name to override the config file')
    parser.add_argument('--prompt', type=str, help='Prompt type to override the config file')
    parser.add_argument('--gpu_ids', type=str, help='Gpu IDs to override the config file')
    parser.add_argument('--box_csv', type=str, default=None)
    parser.add_argument('--box_offset', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_esbct', action='store_true',\
                        help='disable E-SBCT (PAL-only ablation)')
    parser.add_argument('--pal_select', type=str, default=None, choices=['iou', 'gt_dice'])
    args = parser.parse_args()
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
