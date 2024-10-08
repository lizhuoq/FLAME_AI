#!/bin/bash

module load anaconda/2020.11
module load cuda/12.1
source activate climsim

model_name=UNet

# for i in {0..8}
# do
#     python -u run.py \
#         --task_name flame \
#         --is_training -1 \
#         --model_id enc_in_23_target_xi \
#         --model $model_name \
#         --data FLAME \
#         --seq_len 5 \
#         --pred_len 20 \
#         --enc_in 25 \
#         --c_out 1 \
#         --des 'Exp' \
#         --itr 1 \
#         --warmup_epochs 3 \
#         --n_fold $i \
#         --learning_rate 5e-4 \
#         --batch_size 16 \
#         --patience 5 \
#         --train_epochs 20 \
#         --num_workers 0 \
#         --scale 4 \
#         --target_index -1
#     python -u run.py \
#         --task_name flame \
#         --is_training -1 \
#         --model_id enc_in_23_target_xi \
#         --model $model_name \
#         --data FLAME \
#         --seq_len 5 \
#         --pred_len 20 \
#         --enc_in 23 \
#         --c_out 1 \
#         --des 'Exp' \
#         --itr 1 \
#         --warmup_epochs 3 \
#         --n_fold $i \
#         --learning_rate 5e-4 \
#         --batch_size 16 \
#         --patience 5 \
#         --train_epochs 20 \
#         --num_workers 0 \
#         --scale 4 \
#         --target_index -1
# done

for i in {0..8}
do
    python -u run.py \
        --task_name flame \
        --is_training 1 \
        --model_id enc_in_23_target_theta \
        --model $model_name \
        --data FLAME \
        --seq_len 5 \
        --pred_len 20 \
        --enc_in 23 \
        --c_out 1 \
        --des 'Exp' \
        --itr 1 \
        --warmup_epochs 3 \
        --n_fold $i \
        --learning_rate 5e-4 \
        --batch_size 16 \
        --patience 5 \
        --train_epochs 20 \
        --num_workers 0 \
        --scale 4 \
        --target_index 0
done

# python -u run.py \
#     --task_name flame \
#     --is_training 1 \
#     --model_id enc_in_23_target_theta \
#     --model $model_name \
#     --data FLAME \
#     --seq_len 5 \
#     --pred_len 20 \
#     --enc_in 23 \
#     --c_out 1 \
#     --des 'Exp' \
#     --itr 1 \
#     --warmup_epochs 3 \
#     --n_fold $i \
#     --learning_rate 5e-4 \
#     --batch_size 16 \
#     --patience 5 \
#     --train_epochs 20 \
#     --num_workers 0 \
#     --scale 4 \
#     --target_index 0

# python -u run.py \
#     --task_name flame \
#     --is_training 1 \
#     --model_id enc_in_23_target_ustar \
#     --model $model_name \
#     --data FLAME \
#     --seq_len 5 \
#     --pred_len 20 \
#     --enc_in 23 \
#     --c_out 1 \
#     --des 'Exp' \
#     --itr 1 \
#     --warmup_epochs 3 \
#     --n_fold $i \
#     --learning_rate 5e-4 \
#     --batch_size 16 \
#     --patience 5 \
#     --train_epochs 20 \
#     --num_workers 0 \
#     --scale 4 \
#     --target_index 1