#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=UNet

for i in {0..8}
do
    echo ------------------------------------------training fold$i-------------------------------------------------------
    python -u run.py \
        --task_name flame \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96 \
        --model $model_name \
        --data FLAME \
        --features M \
        --seq_len 5 \
        --label_len 48 \
        --pred_len 20 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 38 \
        --dec_in 21 \
        --c_out 1 \
        --d_model 256 \
        --d_ff 75 \
        --top_k 5 \
        --des 'Exp' \
        --itr 1 \
        --warmup_epochs 3 \
        --n_fold $i \
        --learning_rate 5e-4 \
        --batch_size 16 \
        --patience 5 \
        --train_epochs 20 \
        --num_workers 0 \
        --scale 4 
done
