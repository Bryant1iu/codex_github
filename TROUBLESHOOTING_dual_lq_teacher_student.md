# 两阶段训练实现说明（先教师，后蒸馏学生）

你要求的流程是：

1. **阶段一**：先训练 Teacher；
2. **阶段二**：冻结 Teacher，把知识蒸馏到 Student。

本仓库的 `dual_lq_teacher_student_fix.py` 已给出可直接复用的模板函数：

- `teacher_stage_step(...)`：只训练 teacher 分支（`forward_teacher`）
- `student_stage_step(...)`：冻结 teacher 后，训练 student（`recon + distill + sparse`）
- `train_two_stage_template(...)`：完整串联两个阶段

## 推荐执行顺序

- Stage-1（Teacher）:
  - 仅优化 teacher 参数
  - 目标：`L1(pred_t, hq)`

- Stage-2（Student Distill）:
  - 冻结 teacher 参数
  - 优化 student（和你需要训练的重建分支）
  - 目标：`recon + distill + sparse`

## 为什么这样更稳

- teacher 先收敛，student 蒸馏目标更稳定；
- 避免 teacher/student 同时漂移导致 distill 目标震荡；
- 更容易形成 teacher upper-bound 与 student 的性能差异。

## 关键监控

- `stage1/loss_teacher`
- `stage2/loss_recon`
- `stage2/loss_distill`
- `stage2/lambda_distill_eff`
- `debug/pred_gap_l1`
- `debug/comp_gap_l1`

如果 `pred_gap_l1` 长期接近 0，说明 teacher 信息没有有效传递给 student，可增大 `lambda_distill`（配合 ramp）或检查注入是否生效。
