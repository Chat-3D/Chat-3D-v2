which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=53170 # 53170 53171 53173 53177
export MASTER_ADDR=localhost
echo "MASTER_ADDR="$MASTER_ADDR

stage=2
epoch=3
max_txt_len=32
lr=5e-4
dp=0.1
add_scene_token=True
evaluate=False
tag=""
# pretrained_path="outputs/20240302_152031_dp0.1_lr1e-4_sta1_ep10_bs3*4_nonormscale/ckpt_09.pth"
# pretrained_path="outputs/20240302_131430_dp0.1_lr1e-4_sta2_ep10_bs3*4_nonormscale/ckpt_04.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240303_022944_dp0.1_lr2e-4_sta2_ep10_bs3*4_addcaption/ckpt_06.pth"

pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240304_003707_dp0.1_lr2e-4_sta1_ep6_bs3*4_scan2cap25/ckpt_05.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240305_010304_dp0.1_lr5e-4_sta2_ep3_bs3*1_joint_training/ckpt_01.pth"
# pretrained_path="outputs/20240318_211509_dp0.1_lr5e-4_sta1_ep6_bs3*4_scene_dataset_only/ckpt_05.pth"
# pretrained_path=


OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_dp"$dp"_lr"$lr"_sta"$stage"_ep"$epoch"_bs3*4_"$tag"

mkdir -p ${OUTPUT_DIR}
srun \
    --partition=mozi-S1 \
    --gres=gpu:1 \
    --ntasks-per-node=1 \
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
    wandb.enable False
