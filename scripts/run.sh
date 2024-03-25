which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=53172 # 53172 53171 53179 53181
export MASTER_ADDR=localhost
echo "MASTER_ADDR="$MASTER_ADDR

stage=2
epoch=10
max_txt_len=32
lr=5e-6
dp=0.1
add_scene_token=True
evaluate=True
tag="scanrefergrounding_special_trainemb"

debug=true
if [ $debug = "true" ]; then
    enable_wandb=False
    gpu_num=1
else
    enable_wandb=False
    gpu_num=4
fi

pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240325_031151_dp0.1_lr1e-5_sta2_ep10_scenealign_special_trainemb/ckpt_02.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240324_185328_dp0.1_lr5e-6_sta2_ep10_objalign_special_trainemb/ckpt_02.pth"  # old obj align


OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_dp"$dp"_lr"$lr"_sta"$stage"_ep"$epoch"_"$tag"

mkdir -p ${OUTPUT_DIR}
srun \
    --partition=mozi-S1 \
    --gres=gpu:"$gpu_num" \
    --ntasks-per-node="$gpu_num" \
    python tasks/train.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR} \
    model.stage "$stage" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.max_txt_len "$max_txt_len" \
    model.add_scene_token "$add_scene_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    model.mlp_dropout  "$dp" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num"
