#!/bin/bash
#SBATCH --partition=4090
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --exclude=3dimage-20,3dimage-16
#SBATCH -n 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 50g

# custom config
DATA=data
TRAINER=CoOp

DATASET=$1
BACKBONE=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

if [ "$DATASET" == "ALL" ]; then
    DATASETS=(caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101)
else
    DATASETS=( $DATASET )
fi


for SEED in 1 2 3
do
    for SHOTS in  1 2 4 8 16
    do
        for DATASET in ${DATASETS[@]}
        do
            if [ "$DATASET" == "imagenet" ]; then 
                CFG=${BACKBONE}_ep50
            else
                if [ $SHOTS -eq 1 ]; then
                    CFG=${BACKBONE}_ep50
                elif [ $SHOTS -eq 2 ] || [ $SHOTS -eq 4 ]; then
                    CFG=${BACKBONE}_ep100
                elif [ $SHOTS -eq 8 ] || [ $SHOTS -eq 16 ]; then
                    CFG=${BACKBONE}
                else
                    echo "Oops! The number of shots is not supported"
                fi
            fi
            DIR=output/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
            # bash /space/songkun/project/CLIPMI/ckeck.sh $DIR
            if [ -d "$DIR/prompt_learner" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
                :
            else
                bash scripts/coop/run.sh ${DATA} ${SEED} ${TRAINER} ${DATASET} ${CFG} ${DIR} ${NCTX} ${CSC} ${CTP} ${SHOTS}
            fi
        done
    done
done