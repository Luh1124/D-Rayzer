#!/usr/bin/env bash
cd /root/code/depth/E-RayZer

# muon_hybrid: 2D 参数 Muon + 其余 AdamW（需 PyTorch >= 2.11，与 optimizer_factory 一致）
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python training/train.py \
  --config config/data/default.yaml \
  --config config/model/default.yaml \
  --config config/optimizer/muon_hybrid.yaml \
  --config config/train/default.yaml \
  --config config/experiments/inter_rope_rms.yaml \
  --manifest-list /root/data/DL3DV/dl3dv_train_h256/all_manifests.txt \
  --use-dino-curriculum \
  --dino-profile-dir /root/data/DL3DV/dl3dv_train_h256/dino_overlap_profiles \
  --optimizer muon_hybrid \
  --num-views 10 \
  --curriculum-ramp-steps 86000 \
  --batch-size 4 \
  --num-workers 4 \
  --lr 0.0004 \
  --max-steps 100000 \
  --accelerator gpu \
  --devices 6
