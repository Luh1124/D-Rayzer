#!/usr/bin/env bash
set -euo pipefail
cd /root/code/depth/E-RayZer

# 默认栈: config/model,data,optimizer/adamw,train + 下面两个 --config。
# - 论文级设置见 config/train/default.yaml：LR 3K warmup + cosine→0、grad clip/skip、max_effective_updates 等。
# - 停训以「有效更新」为准（152K）；train.py 会把 Trainer.max_steps 抬到至少 cap+slack，避免 skip 时先被 global_step 截断。
# - 论文 global batch 192：此处 6 卡 × 32 = 192；8 卡时可 --devices 8 --batch-size 24。显存不够再略减 batch 或改累积梯度（见 training.grad_accum_steps）。
# - inter_rope_rms.yaml：RoPE + RMSNorm + view_layout（见该文件）。

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python training/train.py \
  --config config/optimizer/muon_hybrid.yaml \
  --config config/experiments/inter_rope_rms.yaml \
  --manifest-list /root/data/DL3DV/dl3dv_train_h256/all_manifests.txt \
  --use-dino-curriculum \
  --dino-profile-dir /root/data/DL3DV/dl3dv_train_h256/dino_overlap_profiles \
  --num-views 10 \
  --curriculum-ramp-steps 86000 \
  --max-effective-updates 152000 \
  --batch-size 32 \
  --num-workers 4 \
  --lr 0.0004 \
  --max-steps 152000 \
  --accelerator gpu \
  --devices 6
