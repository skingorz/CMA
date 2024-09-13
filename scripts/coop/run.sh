#!/bin/bash
#SBATCH --partition=3090
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --exclude=3dimage-20,3dimage-16
#SBATCH -n 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 50g
export CUDA_VISIBLE_DEVICES=0

DATA=$1
SEED=$2
TRAINER=$3
DATASET=$4
CFG=$5
DIR=$6
NCTX=$7
CSC=$8
CTP=${9}
SHOTS=${10}


python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
TRAINER.COOP.N_CTX ${NCTX} \
TRAINER.COOP.CSC ${CSC} \
TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS}

# if [[ "$DATASET" != "imagenet" && "$DATASET" != "sun397" ]]; then
#     echo $(hostname)
#     echo cp and unzip ${DATASET} to /tmp
#     if [ "$DATASET" == "food101" ]; then
#       mkdir -p /tmp/food-101
#       cp /space/songkun/project/CLIPMI/data/food-101.zip /tmp
#       /usr/bin/unzip -q /tmp/food-101.zip -d /tmp/food-101
#       rm /tmp/food-101.zip
#     elif [ "$DATASET" == "caltech101" ]; then
#       mkdir -p /tmp/caltech-101
#       cp /space/songkun/project/CLIPMI/data/caltech-101.zip /tmp
#       /usr/bin/unzip -q /tmp/caltech-101.zip -d /tmp/caltech-101
#       rm /tmp/caltech-101.zip
#     else
#       mkdir -p /tmp/$DATASET
#       cp /space/songkun/project/CLIPMI/data/$DATASET.zip /tmp
#       /usr/bin/unzip -q /tmp/$DATASET.zip -d /tmp/$DATASET
#       rm /tmp/$DATASET.zip
#     fi
#     DATA=/tmp
# fi



