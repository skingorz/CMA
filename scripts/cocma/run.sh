#!/bin/bash
#SBATCH --partition=4090
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --exclude=3dimage-20,3dimage-16
#SBATCH -n 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 50g

DATA=$1
SEED=$2
TRAINER=$3
DATASET=$4
CFG=$5
DIR=$6
SHOTS=$7
lambda=$8


# train
python  train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--lambd ${lambda} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base

# test
SUB=new
MODEL_DIR=${DIR}
LOADEP=10
LOGDIR=${MODEL_DIR}/base2new/test_${SUB}


python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${LOGDIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}