"""
完整可运行：双低质图 teacher-student 两阶段训练脚本

阶段一：Teacher 训练（x1, x2 -> HQ）
阶段二：冻结 Teacher，Student 蒸馏（x1 -> HQ, distill 到 teacher 互补特征）

运行示例：
  # 1) 先训练 teacher
  python dual_lq_teacher_student.py --stage train_teacher --data_root /path/to/data

  # 2) 再蒸馏 student
  python dual_lq_teacher_student.py --stage train_student --data_root /path/to/data \
      --teacher_ckpt ./ckpt_ts/teacher_best.pth

  # 3) 一键两阶段
  python dual_lq_teacher_student.py --stage train_twostage --data_root /path/to/data

  # 4) 评估
  python dual_lq_teacher_student.py --stage eval --data_root /path/to/data
"""

from __future__ import annotations

import os
import math
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ============================================================
# 数据
# ============================================================

class DualLQDataset(Dataset):
    def __init__(self, root: str, split="train", image_size=256, augment=False, seed=42):
        self.lq1_dir = f"{root}/tx8_1"
        self.lq2_dir = f"{root}/tx8_2"
        self.hq_dir = f"{root}/tx128_2"

        all_fn = sorted([
            f for f in os.listdir(self.hq_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
        rng = random.Random(seed)
        rng.shuffle(all_fn)

        n = len(all_fn)
        n_train = int(n * 0.9)
        if split == "train":
            self.filenames = all_fn[:n_train]
        elif split == "val":
            self.filenames = all_fn[n_train:]
        else:
            self.filenames = all_fn

        self.image_size = image_size
        self.augment = augment and split == "train"
        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        print(f"[Dataset] {split}: {len(self.filenames)}{' (aug)' if self.augment else ''}")

    def __len__(self):
        return len(self.filenames)

    def _aug(self, *imgs):
        if random.random() > 0.5:
            imgs = tuple(transforms.functional.hflip(i) for i in imgs)
        if random.random() > 0.5:
            imgs = tuple(transforms.functional.vflip(i) for i in imgs)
        a = random.choice([0, 90, 180, 270])
        if a:
            imgs = tuple(transforms.functional.rotate(i, a) for i in imgs)
        return imgs

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        sz = self.image_size
        lq1 = Image.open(f"{self.lq1_dir}/{fn}").convert("L").resize((sz, sz), Image.BILINEAR)
        lq2 = Image.open(f"{self.lq2_dir}/{fn}").convert("L").resize((sz, sz), Image.BILINEAR)
        hq = Image.open(f"{self.hq_dir}/{fn}").convert("L").resize((sz, sz), Image.BILINEAR)
        if self.augment:
            lq1, lq2, hq = self._aug(lq1, lq2, hq)
        return self.tf(lq1), self.tf(lq2), self.tf(hq)


# ============================================================
# 指标
# ============================================================


def _to_np(t):
    return (t.detach().cpu().float().clamp(-1, 1) * 0.5 + 0.5).numpy()


def calc_psnr(pred, gt):
    p, g = _to_np(pred), _to_np(gt)
    return float(np.mean([peak_signal_noise_ratio(g[i, 0], p[i, 0], data_range=1.0) for i in range(p.shape[0])]))


def calc_ssim(pred, gt):
    p, g = _to_np(pred), _to_np(gt)
    return float(np.mean([structural_similarity(g[i, 0], p[i, 0], data_range=1.0) for i in range(p.shape[0])]))


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


# ============================================================
# 模型基础块
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        skip = self.res(x)
        x = self.down(skip)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.res = ResBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.res(torch.cat([x, skip], dim=1))


class SelfAttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        hd = c // self.heads
        q, k, v = torch.chunk(self.qkv(self.norm(x)), 3, dim=1)
        q = q.reshape(b, self.heads, hd, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.heads, hd, h * w).permute(0, 1, 3, 2)
        v = v.reshape(b, self.heads, hd, h * w).permute(0, 1, 3, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        o = (a @ v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(o)


class PyramidEncoder(nn.Module):
    def __init__(self, base=32, dropout=0.0):
        super().__init__()
        c = base
        self.stem = nn.Conv2d(1, c, 3, padding=1)
        self.d1 = DownBlock(c, c * 2)
        self.d2 = DownBlock(c * 2, c * 4)
        self.d3 = DownBlock(c * 4, c * 8)
        self.mid = nn.Sequential(
            ResBlock(c * 8, c * 8),
            SelfAttnBlock(c * 8),
            ResBlock(c * 8, c * 8),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        s0 = F.silu(self.stem(x))
        x1, sk1 = self.d1(s0)  # H/2, 2C
        x2, sk2 = self.d2(x1)  # H/4, 4C
        x3, sk3 = self.d3(x2)  # H/8, 8C
        bot = self.mid(x3)
        return (x1, x2, x3), (s0, sk1, sk2, sk3), bot


# ============================================================
# 注入
# ============================================================

class ConcatInjector(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.GroupNorm(min(8, ch), ch), nn.SiLU())

    def forward(self, m, c):
        return self.f(torch.cat([m, c], dim=1))


class FiLMInjector(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.g = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.SiLU(), nn.Conv2d(ch, ch, 1))
        self.b = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.SiLU(), nn.Conv2d(ch, ch, 1))
        nn.init.zeros_(self.g[-1].weight)
        nn.init.zeros_(self.g[-1].bias)
        nn.init.zeros_(self.b[-1].weight)
        nn.init.zeros_(self.b[-1].bias)

    def forward(self, m, c):
        return (1 + self.g(c)) * m + self.b(c)


class CrossAttnInjector(nn.Module):
    """只建议用于低分辨率层。"""
    def __init__(self, ch, heads=4):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.nq = nn.GroupNorm(min(8, ch), ch)
        self.nkv = nn.GroupNorm(min(8, ch), ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, m, c):
        b, ch, h, w = m.shape
        hd = ch // self.heads
        q = self.q(self.nq(m)).reshape(b, self.heads, hd, h * w).permute(0, 1, 3, 2)
        k = self.k(self.nkv(c)).reshape(b, self.heads, hd, h * w).permute(0, 1, 3, 2)
        v = self.v(self.nkv(c)).reshape(b, self.heads, hd, h * w).permute(0, 1, 3, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        o = (a @ v).permute(0, 1, 3, 2).reshape(b, ch, h, w)
        return m + self.proj(o)


class GatedInjector(nn.Module):
    def __init__(self, injector, ch):
        super().__init__()
        self.injector = injector
        self.alpha = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, m, c):
        z = self.injector(m, c)
        return m + torch.tanh(self.alpha) * (z - m)


# ============================================================
# Teacher / Student / Recon
# ============================================================


def init_tiny_random_conv(conv: nn.Conv2d, std=1e-3):
    nn.init.normal_(conv.weight, mean=0.0, std=std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class TinyInitHead(nn.Module):
    def __init__(self, ch, std=1e-3):
        super().__init__()
        self.p = nn.Conv2d(ch, ch, 1)
        init_tiny_random_conv(self.p, std)

    def forward(self, x):
        return self.p(x)


class TeacherComp(nn.Module):
    def __init__(self, base=32, dropout=0.0):
        super().__init__()
        c = base
        self.e1 = PyramidEncoder(base=base, dropout=dropout)
        self.e2 = PyramidEncoder(base=base, dropout=dropout)
        self.h1 = TinyInitHead(c * 2)
        self.h2 = TinyInitHead(c * 4)
        self.h3 = TinyInitHead(c * 8)

    def forward(self, x1, x2):
        (a1, a2, a3), _, _ = self.e1(x1)
        (b1, b2, b3), _, _ = self.e2(x2)
        # 互补偏置：显式差分
        c1 = self.h1(torch.cat([a1, b1, b1 - a1], dim=1)[:, :a1.shape[1]])
        c2 = self.h2(torch.cat([a2, b2, b2 - a2], dim=1)[:, :a2.shape[1]])
        c3 = self.h3(torch.cat([a3, b3, b3 - a3], dim=1)[:, :a3.shape[1]])
        return c1, c2, c3


class StudentComp(nn.Module):
    def __init__(self, base=32, dropout=0.0):
        super().__init__()
        c = base
        self.enc = PyramidEncoder(base=base, dropout=dropout)
        self.h1 = TinyInitHead(c * 2)
        self.h2 = TinyInitHead(c * 4)
        self.h3 = TinyInitHead(c * 8)

    def forward(self, x1):
        (f1, f2, f3), _, _ = self.enc(x1)
        return self.h1(f1), self.h2(f2), self.h3(f3)


class ReconNet(nn.Module):
    def __init__(self, base=32, inject_mode="film", dropout=0.0):
        super().__init__()
        c = base
        self.enc = PyramidEncoder(base=base, dropout=dropout)
        self.mid_f = GatedInjector(FiLMInjector(c * 8), c * 8)
        self.u3 = UpBlock(c * 8, c * 8, c * 4)
        self.u2 = UpBlock(c * 4, c * 4, c * 2)
        self.u1 = UpBlock(c * 2, c * 2, c)

        # 防 OOM：crossattn 仅用于 H/4 低分辨率层
        if inject_mode == "crossattn":
            i3 = CrossAttnInjector(c * 4)
            i2 = FiLMInjector(c * 2)
        elif inject_mode == "concat":
            i3 = ConcatInjector(c * 4)
            i2 = ConcatInjector(c * 2)
        else:  # film
            i3 = FiLMInjector(c * 4)
            i2 = FiLMInjector(c * 2)

        self.i3 = GatedInjector(i3, c * 4)
        self.i2 = GatedInjector(i2, c * 2)

        self.out = nn.Sequential(ResBlock(c + c, c), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x1, comp):
        c1, c2, c3 = comp
        _, (s0, sk1, sk2, sk3), bot = self.enc(x1)
        bot = self.mid_f(bot, c3)
        d3 = self.i3(self.u3(bot, sk3), c2)
        d2 = self.i2(self.u2(d3, sk2), c1)
        d1 = self.u1(d2, sk1)
        return torch.tanh(self.out(torch.cat([d1, s0], dim=1)))


class DualTeacherStudent(nn.Module):
    def __init__(self, base=32, inject_mode="film", dropout=0.0):
        super().__init__()
        self.teacher = TeacherComp(base=base, dropout=dropout)
        self.student = StudentComp(base=base, dropout=dropout)
        self.recon = ReconNet(base=base, inject_mode=inject_mode, dropout=dropout)

    def forward_teacher(self, x1, x2):
        comp_t = self.teacher(x1, x2)
        pred_t = self.recon(x1, comp_t)
        return pred_t, comp_t

    def forward_student(self, x1):
        comp_s = self.student(x1)
        pred_s = self.recon(x1, comp_s)
        return pred_s, comp_s


# ============================================================
# 损失/调度
# ============================================================


def mean_l1_gap(a, b):
    return torch.stack([F.l1_loss(x, y) for x, y in zip(a, b)]).mean()


def stable_feature_distill_loss(comp_s, comp_t, clip_per_level=2.0, eps=1e-6):
    tot = 0.0
    for s, t in zip(comp_s, comp_t):
        td = t.detach()
        scale = td.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        sn, tn = s / scale, td / scale
        lv = F.smooth_l1_loss(sn, tn)
        if clip_per_level is not None:
            lv = torch.clamp(lv, max=clip_per_level)
        tot = tot + lv
    return tot / len(comp_s)


def sparsity_loss(comp_s):
    return sum(x.abs().mean() for x in comp_s)


def distill_weight_schedule(epoch, stage1_epochs, target, ramp_epochs=20):
    if epoch <= stage1_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return target
    p = min(1.0, (epoch - stage1_epochs) / float(ramp_epochs))
    return target * p


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def make_scheduler(opt, loader_len, epochs):
    warmup = loader_len * min(3, epochs)
    total = loader_len * epochs

    def f(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@dataclass
class TrainState:
    best_psnr: float = -1.0
    patience_count: int = 0
    step: int = 0


# ============================================================
# 训练阶段1：teacher
# ============================================================


def train_teacher(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args)
    model = DualTeacherStudent(base=args.base_ch, inject_mode=args.inject_mode, dropout=args.dropout).to(device)

    # 阶段一：只训练 teacher + recon
    set_requires_grad(model.teacher, True)
    set_requires_grad(model.recon, True)
    set_requires_grad(model.student, False)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    sch = make_scheduler(opt, len(train_loader), args.teacher_epochs)

    writer = SummaryWriter(log_dir=f"{args.tb_dir}/stage1_teacher")
    os.makedirs(args.save_dir, exist_ok=True)

    state = TrainState()
    for ep in range(1, args.teacher_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, ncols=130, desc=f"[Stage1 Teacher] {ep}/{args.teacher_epochs}")

        for lq1, lq2, hq in pbar:
            lq1, lq2, hq = lq1.to(device), lq2.to(device), hq.to(device)
            pred_t, comp_t = model.forward_teacher(lq1, lq2)
            loss_t = F.l1_loss(pred_t, hq)

            opt.zero_grad()
            loss_t.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

            writer.add_scalar("stage1/loss_teacher", loss_t.item(), state.step)
            writer.add_scalar("stage1/comp_energy", torch.stack([x.abs().mean() for x in comp_t]).mean().item(), state.step)
            pbar.set_postfix(loss=f"{loss_t.item():.4f}")
            state.step += 1

        val_psnr, val_ssim = validate_teacher(model, val_loader, device)
        writer.add_scalar("stage1/val_psnr", val_psnr, ep)
        writer.add_scalar("stage1/val_ssim", val_ssim, ep)
        print(f"[Val Teacher] PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}")

        if val_psnr > state.best_psnr:
            state.best_psnr = val_psnr
            state.patience_count = 0
            torch.save(model.state_dict(), f"{args.save_dir}/teacher_best.pth")
        else:
            state.patience_count += 1
            if state.patience_count >= args.patience:
                print("[Early Stop Stage1]")
                break

    torch.save(model.state_dict(), f"{args.save_dir}/teacher_last.pth")
    writer.close()


# ============================================================
# 训练阶段2：student distill
# ============================================================


def train_student(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args)
    model = DualTeacherStudent(base=args.base_ch, inject_mode=args.inject_mode, dropout=args.dropout).to(device)

    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        model.load_state_dict(torch.load(args.teacher_ckpt, map_location=device), strict=False)
        print(f"[Load teacher_ckpt] {args.teacher_ckpt}")
    else:
        print("[Warn] teacher_ckpt 未提供或不存在，将从当前权重继续。")

    # 阶段二：冻结 teacher，训练 student + recon
    set_requires_grad(model.teacher, False)
    set_requires_grad(model.student, True)
    set_requires_grad(model.recon, True)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    sch = make_scheduler(opt, len(train_loader), args.student_epochs)

    writer = SummaryWriter(log_dir=f"{args.tb_dir}/stage2_student")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/samples", exist_ok=True)

    state = TrainState()
    for ep in range(1, args.student_epochs + 1):
        global_ep = args.teacher_epochs + ep
        model.train()
        pbar = tqdm(train_loader, ncols=140, desc=f"[Stage2 Student] {ep}/{args.student_epochs}")
        for lq1, lq2, hq in pbar:
            lq1, lq2, hq = lq1.to(device), lq2.to(device), hq.to(device)

            pred_s, comp_s = model.forward_student(lq1)
            with torch.no_grad():
                pred_t, comp_t = model.forward_teacher(lq1, lq2)

            loss_recon = F.l1_loss(pred_s, hq)
            loss_dist = stable_feature_distill_loss(comp_s, comp_t, clip_per_level=args.distill_clip_per_level)
            loss_sparse = sparsity_loss(comp_s)
            lam_dist = distill_weight_schedule(global_ep, args.teacher_epochs, args.lambda_distill, args.distill_ramp_epochs)

            loss = args.lambda_recon * loss_recon + lam_dist * loss_dist + args.lambda_sparse * loss_sparse

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

            pred_gap = F.l1_loss(pred_s, pred_t).item()
            comp_gap = mean_l1_gap(comp_s, comp_t).item()
            writer.add_scalar("stage2/loss_total", loss.item(), state.step)
            writer.add_scalar("stage2/loss_recon", loss_recon.item(), state.step)
            writer.add_scalar("stage2/loss_distill", loss_dist.item(), state.step)
            writer.add_scalar("stage2/loss_sparse", loss_sparse.item(), state.step)
            writer.add_scalar("stage2/lambda_distill_eff", lam_dist, state.step)
            writer.add_scalar("debug/pred_gap_l1", pred_gap, state.step)
            writer.add_scalar("debug/comp_gap_l1", comp_gap, state.step)
            writer.add_scalar("debug/comp_energy_s", torch.stack([x.abs().mean() for x in comp_s]).mean().item(), state.step)
            writer.add_scalar("debug/comp_energy_t", torch.stack([x.abs().mean() for x in comp_t]).mean().item(), state.step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{loss_recon.item():.4f}", dist=f"{loss_dist.item():.4f}")
            state.step += 1

        psnr_s, ssim_s, psnr_t = validate_student_teacher(model, val_loader, device)
        writer.add_scalar("stage2/val_psnr_student", psnr_s, ep)
        writer.add_scalar("stage2/val_ssim_student", ssim_s, ep)
        writer.add_scalar("stage2/val_psnr_teacher", psnr_t, ep)
        print(f"[Val Student] student={psnr_s:.2f}/{ssim_s:.4f}  teacher_upper={psnr_t:.2f}")

        if psnr_s > state.best_psnr:
            state.best_psnr = psnr_s
            state.patience_count = 0
            torch.save(model.state_dict(), f"{args.save_dir}/student_best.pth")
        else:
            state.patience_count += 1
            if state.patience_count >= args.patience:
                print("[Early Stop Stage2]")
                break

        if ep % args.save_every == 0 or ep == 1:
            save_vis(model, val_loader, device, ep, args)

    torch.save(model.state_dict(), f"{args.save_dir}/student_last.pth")
    writer.close()


# ============================================================
# 两阶段一键
# ============================================================


def train_twostage(args):
    print("[TwoStage] Stage1 teacher -> Stage2 student distill")
    train_teacher(args)
    args.teacher_ckpt = f"{args.save_dir}/teacher_best.pth"
    train_student(args)


# ============================================================
# 验证/评估
# ============================================================

@torch.no_grad()
def validate_teacher(model, loader, device):
    model.eval()
    ps, ss, n = 0.0, 0.0, 0
    for lq1, lq2, hq in loader:
        lq1, lq2, hq = lq1.to(device), lq2.to(device), hq.to(device)
        pred, _ = model.forward_teacher(lq1, lq2)
        b = lq1.size(0)
        ps += calc_psnr(pred, hq) * b
        ss += calc_ssim(pred, hq) * b
        n += b
    return ps / n, ss / n


@torch.no_grad()
def validate_student_teacher(model, loader, device):
    model.eval()
    ps_s, ss_s, ps_t, n = 0.0, 0.0, 0.0, 0
    for lq1, lq2, hq in loader:
        lq1, lq2, hq = lq1.to(device), lq2.to(device), hq.to(device)
        pred_s, _ = model.forward_student(lq1)
        pred_t, _ = model.forward_teacher(lq1, lq2)
        b = lq1.size(0)
        ps_s += calc_psnr(pred_s, hq) * b
        ss_s += calc_ssim(pred_s, hq) * b
        ps_t += calc_psnr(pred_t, hq) * b
        n += b
    return ps_s / n, ss_s / n, ps_t / n


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_loaders(args)

    ckpt = args.eval_ckpt if args.eval_ckpt else f"{args.save_dir}/student_best.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    model = DualTeacherStudent(base=args.base_ch, inject_mode=args.inject_mode, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    ps_s, ss_s, ps_t = validate_student_teacher(model, val_loader, device)

    print("=" * 66)
    print(f"student: PSNR={ps_s:.2f} SSIM={ss_s:.4f}")
    print(f"teacher upper: PSNR={ps_t:.2f}")
    print("=" * 66)


# ============================================================
# 可视化 & loader
# ============================================================

@torch.no_grad()
def save_vis(model, loader, device, epoch, args):
    model.eval()
    lq1, lq2, hq = next(iter(loader))
    lq1, lq2, hq = lq1.to(device), lq2.to(device), hq.to(device)
    nb = min(4, lq1.size(0))
    ps, _ = model.forward_student(lq1[:nb])
    pt, _ = model.forward_teacher(lq1[:nb], lq2[:nb])
    grid = torch.cat([denorm(lq1[:nb]), denorm(ps), denorm(pt), denorm(hq[:nb])], dim=0)
    save_image(grid, f"{args.save_dir}/samples/ep{epoch:04d}.png", nrow=nb)


def build_loaders(args):
    val_root = args.val_data_root if args.val_data_root else args.data_root
    train_loader = DataLoader(
        DualLQDataset(args.data_root, "train", args.image_size, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        DualLQDataset(val_root, "val", args.image_size, augment=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Dual LQ Teacher-Student Two-Stage Training")
    p.add_argument("--stage", type=str, default="train_twostage",
                   choices=["train_teacher", "train_student", "train_twostage", "eval"])
    p.add_argument("--inject_mode", type=str, default="film", choices=["concat", "film", "crossattn"])

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--val_data_root", type=str, default="")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--teacher_epochs", type=int, default=40)
    p.add_argument("--student_epochs", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=25)

    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_distill", type=float, default=0.1)
    p.add_argument("--lambda_sparse", type=float, default=1e-4)
    p.add_argument("--distill_ramp_epochs", type=int, default=20)
    p.add_argument("--distill_clip_per_level", type=float, default=2.0)

    p.add_argument("--teacher_ckpt", type=str, default="")
    p.add_argument("--eval_ckpt", type=str, default="")

    p.add_argument("--save_dir", type=str, default="./ckpt_ts")
    p.add_argument("--tb_dir", type=str, default="./runs/ts")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.stage == "train_teacher":
        train_teacher(args)
    elif args.stage == "train_student":
        train_student(args)
    elif args.stage == "train_twostage":
        train_twostage(args)
    elif args.stage == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
