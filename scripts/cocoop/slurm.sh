DATASETS=(caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101)
SEED=1
LAMBDAS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

for DATASET in ${DATASETS[@]}
do
    for LAMBDA in ${LAMBDAS[@]}
    do
        bash scripts/cocoop/base2new_train.sh $DATASET $SEED $LAMBDA
    done
done