#!/bin/bash

# custom config
DATA=data
TRAINER=CoCoOp  # (CoCoOp CoCoOpMME CoCoOpMME_CLASSWISE)
# TRAINER=CoOp

DATASET=$1
SEED=$2
lambda=$3

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16

DIR=output/${TRAINER}/lambda${lambda}/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    bash scripts/cocoop/run.sh $DATA ${SEED} ${TRAINER} ${DATASET} ${CFG} ${DIR} ${SHOTS} ${lambda}
fi
