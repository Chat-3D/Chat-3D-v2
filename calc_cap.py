import json
import sys

outputs = json.load(open("/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240307_045334_dp0.1_lr5e-4_sta2_ep3_bs3*1_caption50/preds_epoch2_step6456.json", "r"))
# outputs = json.load(open("outputs/2023-11-24-192539_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign_scanqa/preds_epoch2_step33928.json", "r"))

annos = json.load(open("annotations/scanrefer_mask3d_val_stage2_caption_iou0.json"))
# outputs2 = json.load(open("/root/scene-LLaMA/video_chat/video_chat/outputs/2023-09-17-215602_dp0.1_lr5e-5_sta3_ep3/preds_epoch-1_step0.json", "r"))
# outputs.extend(outputs2)
preds = {}
targets = {}

print(len(outputs))
ref_lens = 0
# acc = 0
# max_len = 40

for anno in annos:
    item_id = f"{anno['scene_id']}_{anno['obj_id']}"
    targets[item_id] = [{"caption": "sos " + caption + " eos"} for caption in anno["ref_captions"]]


for i, output in enumerate(outputs):
    item_id = f"{output['scene_id']}_{output['obj_id']}"
    pred = output["pred"]
    if ": " in pred:
        pred = pred.split(": ")[1].strip()
    preds[item_id] = [{"caption": "sos " + pred + " eos"}]

for k in targets.keys():
    if k not in preds:
        preds[k] = [{"caption": "sos eos"}]
# breakpoint()


pred_keys = set(preds.keys())
target_keys = set(targets.keys())
missing_keys = target_keys - pred_keys

for k in missing_keys:
    del targets[k]
    # preds[k] = [{"caption": "sos eos"}]


# print(ref_lens)
print(len(preds))

# print(preds[:1])
# print(targets[:1])

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# from nltk.translate.bleu_score import sentence_bleu
# score = sentence_bleu(targets[:5], preds[:5], weights=[1.])
# print(score)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    # (Spice(), "SPICE")
]
# scorers = [
#     (capblue.Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#     (capmeteor.Meteor(), "METEOR"),
#     (caprouge.Rouge(), "ROUGE_L"),
#     (capcider.Cider(), "CIDEr")
# ]

tokenizer = PTBTokenizer()
targets = tokenizer.tokenize(targets)
preds = tokenizer.tokenize(preds)

# print(targets)
# print(preds)


val_results = {}

for scorer, method in scorers:
    eprint('computing %s score...'%(scorer.method()))
    score, scores = scorer.compute_score(targets, preds)
    if type(method) == list:
        for sc, scs, m in zip(score, scores, method):
            print("%s: %0.3f"%(m, sc*100))
            val_results[m] = sc
            # rlt[m]=sc*100
    else:
        print("%s: %0.3f"%(method, score*100))
        val_results[method] = score
        # rlt[method]=score*100

for k, v in val_results.items():
    print(f"{k}: {v}")
