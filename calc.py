import json
import sys
replace_list = [["10", "ten"], ["12", "twelve"], ["1", "one"], ["2", "two"], ["3", "three"], ["4", "four"], ["5", "five"],
                ["6", "six"], ["7", "seven"], ["8", "eight"], ["9", "nine"]]
outputs = json.load(open("outputs/2023-11-10-210510_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scenealign_scanqa/preds_epoch2_step17070.json", "r"))
# outputs2 = json.load(open("/root/scene-LLaMA/video_chat/video_chat/outputs/2023-09-17-215602_dp0.1_lr5e-5_sta3_ep3/preds_epoch-1_step0.json", "r"))
# outputs.extend(outputs2)
preds = {}
targets = {}

print(len(outputs))
ref_lens = 0
acc = 0
# max_len = 40

for i, output in enumerate(outputs):
    item_id = f"{output['scene_id']}_{output['obj_id']}_{output['qid']}_{i}"
    pred = output["pred"]
    ref_captions = output["ref_captions"]
    if ": " in pred:
        pred = pred.split(": ")[1].strip()
    if len(pred) > 0 and pred[-1] == ".":
        pred = pred[:-1]
    if len(pred) > 0:
        pred = pred[0].lower() + pred[1:]
    # pred = pred[:max_len]
    # output["ref_captions"][0] = output["ref_captions"][0][:max_len]
    # if output["ref_captions"][0][-1] == ".":
    #     output["ref_captions"][0] = output["ref_captions"][0][:-1]
    for j in range(len(ref_captions)):
        for rep in replace_list:
            if ref_captions[j].startswith(rep[1]):
                ref_captions[j] = ref_captions[j].replace(rep[1], rep[0])
    for rep in replace_list:
        if pred.startswith(rep[1]):
            pred = pred.replace(rep[1], rep[0], 1)
    output["pred"] = pred
    output["ref_captions"] = ref_captions
    if pred in ref_captions:
        acc += 1
            # print(f"Pred: {pred} | Target: {ref_captions[0]}")
        # if ref_captions[j][-1] != ".":
        #     ref_captions[j] += "."
    preds[item_id] = [{"caption": pred}]
    targets[item_id] = [{"caption": caption} for caption in ref_captions]
    # print(targets[item_id][0])
    ref_lens += len(targets[item_id][0])
    # preds.append(output["pred"])
    # targets.append(output["ref_captions"])

# with open("scanqa_sota_result.json", "w") as f:
#     json.dump(outputs, f, indent=4)

# preds["1"] = [" this is the lamp on the round table. it is by the corner."]
# targets["1"] = ["this  is the lamp on the round table.  it is by the corner."]

# print(ref_lens)
print(len(preds))
print("EM: ", float(acc) / len(preds))

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
# from pycocoevalcap.spice.spice import Spice

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
    (Spice(), "SPICE")
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
