# ========================= data ==========================
anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d"
segmentor = "mask3d"
feat_file = f"{anno_root}/scannet_{pc_encoder}_feats.pt"
# attribute_file = f"{anno_root}/scannet_attributes.json"
train_file_s1 = [
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scanrefer_train_stage1.json",
    # ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/nr3d_train_stage1.json",
    # ],
    [
        feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scannet_train_stage1.json",
    ],
    # [
    #     f"{anno_root}/objaverse_{pc_encoder}_feats.pt",
    #     f"{anno_root}/objaverse_attributes.json",
    #     f"{anno_root}/objaverse_stage1.json"
    # ]
]
train_file_s2 = [
    [
        feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanrefer_train_stage2_caption.json",
    ],
    [
        feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scannet_train_stage2_caption.json",
    ],
    [
        feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/nr3d_train_stage2_caption.json"
    ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scanqa_train_stage2_objxx.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_train_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_train_stage2_caption_iou50.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_train_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_train_stage2_caption_iou25.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/sr3d_train_stage2_objxx.json"
    # # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scene_align_train.json",
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/obj_align_train.json",
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scanqa_train_stage2_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/nr3d_train_stage2_grounding_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scanrefer_train_stage2_grounding_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/sr3d_train_stage2_grounding_new.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_train_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_train_stage2_grounding_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/nr3d_train_stage2_multichoice0.01.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scene_dataset_train_stage2.json"
    # ]
]
val_file_s2 = [
    [
        feat_file,
        f"{anno_root}/scannet_val_attributes.pt",
        f"{anno_root}/scanrefer_val_stage2_caption100.json"
    ],
    [
        feat_file,
        f"{anno_root}/scannet_val_attributes.pt",
        f"{anno_root}/nr3d_val_stage2_caption100.json"
    ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanqa_val_stage2_objxx100.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_val_stage2_caption_iou25.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/stage2_val400.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/nr3d_val_stage2_objxx.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scene_align_val.json",
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/obj_align_val.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanqa_val_stage2_objxx.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanqa_val_stage2_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/sr3d_val_stage2_grounding_new.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_val_stage2_grounding_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/nr3d_val_stage2_multichoice0.01.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scene_dataset_val_stage2.json"
    # ]
]
train_file_s3 = [
    [
        feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanqa_train_stage3.json",
        1
    ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/scanrefer_train_conversation.json",
    #     3
    # ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/scanrefer_train_detail.json",
    #     1
    # ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/nr3d_train_tf.json",
    #     1
    # ]
]
val_file_s1 = [
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanrefer_val_stage1.json",
    # ],
    [
        feat_file,
        f"{anno_root}/scannet_val_attributes.pt",
        f"{anno_root}/scannet_val_stage1.json",
    ]
]
val_file_s3 = [
    [
        feat_file,
        f"{anno_root}/scannet_val_attributes.pt",
        f"{anno_root}/scanqa_val_predobj.json"
    ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/scanrefer_val_conversation100.json"
    # ],
]


test_types = []
num_workers = 32

# ========================= input ==========================
s1_batch_size = 64
s2_batch_size = 24
s3_batch_size = 1
# max_txt_l = 32

pre_text = False


# ========================= model ==========================
model = dict(
    llama_model_path="model/vicuna-7b-v0",
    input_dim=1024 if pc_encoder == "uni3d" else 512,
    attr_dim=512,
    encoder_num_layers=1,
    mlp_dropout=0.1,
    low_resource=False,
    system_path="prompts/system.txt",
    prompt_template="\n### Human: {}\n### Assistant: ",
    max_txt_len=512,
    end_sym="\n###",
    stage=3,
    add_scene_token=True,
    debug=False,
    obj_norm_scale=200,
    scene_norm_scale=50,
    grad_scale=1,
)

optimizer = dict(
    opt="adamW",
    lr=5e-3,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=True,
        module_names=["module.llama_model", "module.relation_module"],
        lr=[1e-5, 1e-5],
        wd=[0.02, 0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.2)

evaluate = False
deep_fusion = False

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="huanghaifeng",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="Scene-LLM",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
# eval_freq = 500
seed = 42

save_latest = False
do_save = True
auto_resume = True
pretrained_path = ""
