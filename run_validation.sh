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
python validation.py \
s3://juxiaoliang/dataset/waymo/waymo_sf_processed/valid \
/mnt/petrelfs/juxiaoliang/project/sf/FastFlow3D/config.yaml \
--model_path /mnt/petrelfs/juxiaoliang/project/sf/FastFlow3D/lightning_logs/version_4590154/checkpoints/last.ckpt