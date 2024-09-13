
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
bash scripts/coop/main.sh ALL rn50 end 16 1 False
