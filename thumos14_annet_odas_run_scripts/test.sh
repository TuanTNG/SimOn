#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

python main.py \
    --output_dir "models/thumos_anet_numlayer3_decoder_layer4_enc_stride_1_kv7" \
    --pickle_root "/data/thumos14_anet" \
    --dim_feature 3072 \
    --history_desision 6 \
    --history_feature 6 \
    --epochs 17 --lr_drop 3 --gamma 0.5 \
    --eval --resume "models/thumos_anet_numlayer3_decoder_layer4_enc_stride_1_kv7/checkpoint0016.pth"\
