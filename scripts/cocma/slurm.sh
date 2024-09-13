DATASETS=(imagenet caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101)
# SEED=1
LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

LAMBDA=$1

export CUDA_VISIBLE_DEVICES=$2

for SEED in 1 2 3
do
    for DATASET in ${DATASETS[@]}
    do
        bash scripts/cocma/base2new_train.sh $DATASET $SEED $LAMBDA
    done
done

# bash scripts/cocma/slurm.sh 0.1 1