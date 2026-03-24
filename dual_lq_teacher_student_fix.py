"""
Drop-in fixes for dual_lq_teacher_student training where:
  - loss_distill == 0
  - loss_sparse  == 0

This module provides minimal code you can copy into your existing
`dual_lq_teacher_student.py`:
  1) avoid dual-zero lock in teacher/student complement heads,
  2) enforce explicit warmup -> joint phase switch,
  3) add debug scalars to verify comp branches are alive.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Initialization helpers
# ============================================================

def init_tiny_random_conv(conv: nn.Conv2d, std: float = 1e-3) -> None:
    """Tiny-random init to prevent exact-zero dead start."""
    nn.init.normal_(conv.weight, mean=0.0, std=std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class TinyInitHead(nn.Module):
    """
    Replacement for strict ZeroConv when distill/sparse stays at zero.
    Keeps initialization very small, but not identically zero.
    """

    def __init__(self, channels: int, std: float = 1e-3):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        init_tiny_random_conv(self.proj, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============================================================
# 2) Example: student/teacher complement heads
# ============================================================

class StudentComplementHead(nn.Module):
    """
    Suggested student head: remove strict ZeroConv at output.
    """

    def __init__(self, c1: int, c2: int, c3: int, std: float = 1e-3):
        super().__init__()
        self.pred_l1 = TinyInitHead(c1, std=std)
        self.pred_l2 = TinyInitHead(c2, std=std)
        self.pred_l3 = TinyInitHead(c3, std=std)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor):
        return self.pred_l1(f1), self.pred_l2(f2), self.pred_l3(f3)


class TeacherComplementHead(nn.Module):
    """
    Keep teacher learnable and non-zero start as well.
    """

    def __init__(self, c1: int, c2: int, c3: int, std: float = 1e-3):
        super().__init__()
        self.out_l1 = TinyInitHead(c1, std=std)
        self.out_l2 = TinyInitHead(c2, std=std)
        self.out_l3 = TinyInitHead(c3, std=std)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        return self.out_l1(x1), self.out_l2(x2), self.out_l3(x3)


# ============================================================
# 3) Loss + logging utilities
# ============================================================



def mean_l1_gap(feats_a, feats_b) -> torch.Tensor:
    """Average L1 gap over multi-level feature tuples."""
    gaps = [F.l1_loss(a, b) for a, b in zip(feats_a, feats_b)]
    return torch.stack(gaps).mean()


def branch_collapse_diagnostics(pred_s: torch.Tensor, pred_t: torch.Tensor, comp_s, comp_t):
    """
    Returns diagnostics to detect teacher/student collapse:
      - pred_gap_l1: if ~0, both branches produce near-identical outputs
      - comp_gap_l1: if ~0, complement features are near-identical
      - comp_energy_s/t: branch magnitude
    """
    pred_gap = F.l1_loss(pred_s, pred_t)
    comp_gap = mean_l1_gap(comp_s, comp_t)
    comp_energy_s = torch.stack([x.abs().mean() for x in comp_s]).mean()
    comp_energy_t = torch.stack([x.abs().mean() for x in comp_t]).mean()
    return {
        "pred_gap_l1": pred_gap,
        "comp_gap_l1": comp_gap,
        "comp_energy_s": comp_energy_s,
        "comp_energy_t": comp_energy_t,
    }

def feature_distill_loss(comp_s, comp_t, weights=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Distill student to teacher (teacher detached by design).
    """
    loss = 0.0
    for w, s, t in zip(weights, comp_s, comp_t):
        loss = loss + w * F.l1_loss(s, t.detach())
    return loss




def stable_feature_distill_loss(
    comp_s,
    comp_t,
    weights=(1.0, 1.0, 1.0),
    eps: float = 1e-6,
    clip_per_level: float | None = 2.0,
) -> torch.Tensor:
    """
    Scale-stable distillation:
      - normalize by teacher feature magnitude (per sample)
      - use smooth L1 instead of plain L1
      - optional per-level clipping to prevent explosion
    """
    total = 0.0
    for w, s, t in zip(weights, comp_s, comp_t):
        td = t.detach()
        scale = td.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        sn = s / scale
        tn = td / scale
        lv = F.smooth_l1_loss(sn, tn)
        if clip_per_level is not None:
            lv = torch.clamp(lv, max=clip_per_level)
        total = total + w * lv
    return total


def distill_weight_schedule(epoch: int, warmup_epochs: int, target_weight: float, ramp_epochs: int = 20) -> float:
    """Ramp distill weight after warmup to avoid sudden instability."""
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return target_weight
    p = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    return target_weight * p

def sparsity_loss(comp_s) -> torch.Tensor:
    return sum(feat.abs().mean() for feat in comp_s)


def get_train_phase(epoch: int, warmup_epochs: int) -> str:
    return "warmup" if epoch <= warmup_epochs else "joint"


def log_comp_debug(writer, global_step: int, comp_s, comp_t, phase: str, pred_s=None, pred_t=None) -> None:
    """
    Add runtime visibility for dead comp branches.
    """
    writer.add_scalar("debug/is_joint", float(phase == "joint"), global_step)

    # mean over all levels (safe scalar view)
    s_mean = torch.stack([x.abs().mean() for x in comp_s]).mean().item()
    t_mean = torch.stack([x.abs().mean() for x in comp_t]).mean().item()

    writer.add_scalar("debug/comp_s_abs_mean", s_mean, global_step)
    writer.add_scalar("debug/comp_t_abs_mean", t_mean, global_step)

    if pred_s is not None and pred_t is not None:
        diag = branch_collapse_diagnostics(pred_s, pred_t, comp_s, comp_t)
        writer.add_scalar("debug/pred_gap_l1", diag["pred_gap_l1"].item(), global_step)
        writer.add_scalar("debug/comp_gap_l1", diag["comp_gap_l1"].item(), global_step)
        writer.add_scalar("debug/comp_energy_s", diag["comp_energy_s"].item(), global_step)
        writer.add_scalar("debug/comp_energy_t", diag["comp_energy_t"].item(), global_step)


# ============================================================
# 4) Copy-paste training step template
# ============================================================

def training_step_template(
    *,
    epoch: int,
    warmup_epochs: int,
    model,
    batch,
    optimizer,
    writer,
    global_step: int,
    lambda_recon: float,
    lambda_distill: float,
    lambda_sparse: float,
    lambda_teacher: float,
    distill_ramp_epochs: int = 20,
    distill_clip_per_level: float | None = 2.0,
):
    """
    Expected model interface:
      pred_s, comp_s = model.forward_student(lq1)
      pred_t, comp_t = model.forward_teacher(lq1, lq2)

    This function illustrates a safe loss wiring that avoids silent zero lock.
    """
    lq1, lq2, hq = batch
    phase = get_train_phase(epoch, warmup_epochs)

    optimizer.zero_grad(set_to_none=True)

    if phase == "warmup":
        # warmup: only reconstruction on student path
        pred_s, comp_s = model.forward_student(lq1)
        with torch.no_grad():
            pred_t, comp_t = model.forward_teacher(lq1, lq2)

        loss_recon = F.l1_loss(pred_s, hq)
        loss_distill = torch.zeros((), device=loss_recon.device)
        loss_sparse = torch.zeros((), device=loss_recon.device)
        loss_teacher = torch.zeros((), device=loss_recon.device)
        lambda_distill_eff = 0.0
        loss_total = lambda_recon * loss_recon

    else:
        pred_s, comp_s = model.forward_student(lq1)
        pred_t, comp_t = model.forward_teacher(lq1, lq2)

        loss_recon = F.l1_loss(pred_s, hq)
        loss_distill = stable_feature_distill_loss(comp_s, comp_t, clip_per_level=distill_clip_per_level)
        loss_sparse = sparsity_loss(comp_s)
        loss_teacher = F.l1_loss(pred_t, hq)

        lambda_distill_eff = distill_weight_schedule(
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            target_weight=lambda_distill,
            ramp_epochs=distill_ramp_epochs,
        )

        loss_total = (
            lambda_recon * loss_recon
            + lambda_distill_eff * loss_distill
            + lambda_sparse * loss_sparse
            + lambda_teacher * loss_teacher
        )

    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    writer.add_scalar("train/loss_total", loss_total.item(), global_step)
    writer.add_scalar("train/loss_recon", loss_recon.item(), global_step)
    writer.add_scalar("train/loss_distill", loss_distill.item(), global_step)
    writer.add_scalar("train/loss_sparse", loss_sparse.item(), global_step)
    writer.add_scalar("train/loss_teacher", loss_teacher.item(), global_step)
    writer.add_scalar("train/lambda_distill_eff", float(lambda_distill_eff), global_step)
    log_comp_debug(writer, global_step, comp_s, comp_t, phase, pred_s=pred_s, pred_t=pred_t)

    return {
        "phase": phase,
        "loss_total": float(loss_total.item()),
        "loss_recon": float(loss_recon.item()),
        "loss_distill": float(loss_distill.item()),
        "loss_sparse": float(loss_sparse.item()),
        "loss_teacher": float(loss_teacher.item()),
        "lambda_distill_eff": float(lambda_distill_eff),
        "pred_gap_l1": float(F.l1_loss(pred_s, pred_t).item()),
        "comp_gap_l1": float(mean_l1_gap(comp_s, comp_t).item()),
    }


# ============================================================
# 5) Two-stage training (requested):
#    Stage-1 train teacher, Stage-2 distill to student
# ============================================================

def freeze_module(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def teacher_stage_step(
    *,
    model,
    batch,
    optimizer,
    writer,
    global_step: int,
    lambda_teacher_recon: float = 1.0,
):
    """
    Stage-1: train teacher branch only.

    Expected model interface:
      pred_t, comp_t = model.forward_teacher(lq1, lq2)
    """
    lq1, lq2, hq = batch

    optimizer.zero_grad(set_to_none=True)
    pred_t, comp_t = model.forward_teacher(lq1, lq2)

    loss_teacher = F.l1_loss(pred_t, hq)
    loss_total = lambda_teacher_recon * loss_teacher

    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    comp_energy_t = torch.stack([x.abs().mean() for x in comp_t]).mean()

    writer.add_scalar("stage1/loss_teacher", loss_teacher.item(), global_step)
    writer.add_scalar("stage1/loss_total", loss_total.item(), global_step)
    writer.add_scalar("stage1/comp_energy_t", comp_energy_t.item(), global_step)

    return {
        "loss_teacher": float(loss_teacher.item()),
        "loss_total": float(loss_total.item()),
        "comp_energy_t": float(comp_energy_t.item()),
    }


def student_stage_step(
    *,
    epoch: int,
    distill_start_epoch: int,
    model,
    batch,
    optimizer,
    writer,
    global_step: int,
    lambda_recon: float = 1.0,
    lambda_distill: float = 0.1,
    lambda_sparse: float = 1e-4,
    distill_ramp_epochs: int = 20,
    distill_clip_per_level: float | None = 2.0,
):
    """
    Stage-2: freeze teacher and distill to student.

    Expected model interface:
      pred_s, comp_s = model.forward_student(lq1)
      pred_t, comp_t = model.forward_teacher(lq1, lq2)
    """
    lq1, lq2, hq = batch

    optimizer.zero_grad(set_to_none=True)

    pred_s, comp_s = model.forward_student(lq1)
    with torch.no_grad():
        pred_t, comp_t = model.forward_teacher(lq1, lq2)

    loss_recon = F.l1_loss(pred_s, hq)
    loss_distill = stable_feature_distill_loss(
        comp_s, comp_t, clip_per_level=distill_clip_per_level
    )
    loss_sparse = sparsity_loss(comp_s)

    lambda_distill_eff = distill_weight_schedule(
        epoch=epoch,
        warmup_epochs=distill_start_epoch,
        target_weight=lambda_distill,
        ramp_epochs=distill_ramp_epochs,
    )

    loss_total = (
        lambda_recon * loss_recon
        + lambda_distill_eff * loss_distill
        + lambda_sparse * loss_sparse
    )

    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    writer.add_scalar("stage2/loss_total", loss_total.item(), global_step)
    writer.add_scalar("stage2/loss_recon", loss_recon.item(), global_step)
    writer.add_scalar("stage2/loss_distill", loss_distill.item(), global_step)
    writer.add_scalar("stage2/loss_sparse", loss_sparse.item(), global_step)
    writer.add_scalar("stage2/lambda_distill_eff", float(lambda_distill_eff), global_step)

    log_comp_debug(
        writer,
        global_step,
        comp_s,
        comp_t,
        phase="distill",
        pred_s=pred_s,
        pred_t=pred_t,
    )

    return {
        "loss_total": float(loss_total.item()),
        "loss_recon": float(loss_recon.item()),
        "loss_distill": float(loss_distill.item()),
        "loss_sparse": float(loss_sparse.item()),
        "lambda_distill_eff": float(lambda_distill_eff),
        "pred_gap_l1": float(F.l1_loss(pred_s, pred_t).item()),
        "comp_gap_l1": float(mean_l1_gap(comp_s, comp_t).item()),
    }


def train_two_stage_template(
    *,
    model,
    teacher_optimizer,
    student_optimizer,
    teacher_loader,
    student_loader,
    writer,
    teacher_epochs: int,
    student_epochs: int,
):
    """
    End-to-end two-stage schedule:

    Stage-1 (teacher pretrain):
      - train teacher path only
      - freeze student path

    Stage-2 (student distill):
      - freeze teacher path
      - train student path with recon + distill
    """
    global_step = 0

    # ---- Stage-1: teacher ----
    freeze_module(model, True)
    # If your model has explicit submodules, replace by:
    # freeze_module(model.teacher, True)
    # freeze_module(model.student, False)
    # freeze_module(model.recon, True)

    for ep in range(1, teacher_epochs + 1):
        for batch in teacher_loader:
            teacher_stage_step(
                model=model,
                batch=batch,
                optimizer=teacher_optimizer,
                writer=writer,
                global_step=global_step,
            )
            global_step += 1

    # ---- Stage-2: student distill ----
    freeze_module(model, True)
    # If your model has explicit submodules, replace by:
    # freeze_module(model.teacher, False)
    # freeze_module(model.student, True)
    # freeze_module(model.recon, True)

    for ep in range(1, student_epochs + 1):
        epoch_id = teacher_epochs + ep
        for batch in student_loader:
            student_stage_step(
                epoch=epoch_id,
                distill_start_epoch=teacher_epochs,
                model=model,
                batch=batch,
                optimizer=student_optimizer,
                writer=writer,
                global_step=global_step,
            )
            global_step += 1


__all__ = [
    "TinyInitHead",
    "StudentComplementHead",
    "TeacherComplementHead",
    "mean_l1_gap",
    "branch_collapse_diagnostics",
    "feature_distill_loss",
    "stable_feature_distill_loss",
    "distill_weight_schedule",
    "sparsity_loss",
    "get_train_phase",
    "log_comp_debug",
    "training_step_template",
    "freeze_module",
    "teacher_stage_step",
    "student_stage_step",
    "train_two_stage_template",
]
