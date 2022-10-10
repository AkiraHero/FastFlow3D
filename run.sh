#!/bin/bash
set -e

# if [[ "$VIRTUAL_ENV" == "" ]]
# then
#         echo "not inside virtual env!"
#         exit 1
# fi

echo "Make sure to log to the correct wandb project!"
export WANDB_API_KEY=749dd12d56bf890c0d38553e41942a2c8f434541
export PYTHONPATH="/mnt/petrelfs/juxiaoliang/project/petrel_utils":${PYTHONPATH}
# Run a training run
# Batch size is PER GPU
# Num workers is likely PER GPU and each gpu has another process as well
python train.py \
s3://juxiaoliang/dataset/waymo/waymo_sf_processed \
FastFlowNet_batchSize_16_lr_0.0001_BN_8_2 \
--accelerator cuda \
--sync_batchnorm True \
--batch_size 32 \
--gpus 4 \
--num_workers 4 \
--wandb_enable False \
--wandb_project fastflow3d \
--wandb_entity akira95 \
--learning_rate 0.0001 \
--disable_ddp_unused_check False