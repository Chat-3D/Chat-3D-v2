import json
import random
import nltk
from nltk.tokenize import sent_tokenize
import jsonlines

# nltk.download("punkt")

dataset = "scanrefer"
split = "train"
anno_file = f"annotations/{dataset}_{split}_stage2_objxx.json"

annos = json.load(open(anno_file, "r"))


# with open("prompts/concise_description_objxx.txt", 'r') as f:
#     prompt_candidates = f.read().splitlines()

# import tqdm
# for i in tqdm.tqdm(range(len(annos))):
#     # for j in range(len(annos[i]["ref_captions"])):
#     #     annos[i]["ref_captions"][j] = annos[i]["ref_captions"][j].replace("  ", " ").replace(" ,", ",").replace(" .", ".").strip()
#     obj_id = annos[i]["obj_id"]
#     # del(annos[i]["obj_id"])
#     annos[i]["prompt"] = random.choice(prompt_candidates).replace("<id>", f"{obj_id:02}")
#     if split == "train":
#         sentences = sent_tokenize(annos[i]["caption"])
#         altered_sentences = [' '.join(sentence.split()) for sentence in sentences]
#         annos[i]["caption"] = ' '.join([sentence.capitalize() for sentence in altered_sentences])
#         if annos[i]["caption"][-1] != ".":
#             annos[i]["caption"] += "."
#     else:
#         for j in range(len(annos[i]["ref_captions"])):
#             sentences = sent_tokenize(annos[i]["ref_captions"][j])
#             altered_sentences = [' '.join(sentence.split()) for sentence in sentences]
#             annos[i]["ref_captions"][j] = ' '.join([sentence.capitalize() for sentence in altered_sentences])
#             if annos[i]["ref_captions"][j][-1] != ".":
#                 annos[i]["ref_captions"][j] += "."
#
#
# with open(f"annotations/{dataset}_{split}_stage2_objxx.json", "w") as f:
#     json.dump(annos, f, indent=4)

# with open("prompts/grounding_prompts.txt", 'r') as f:
#     prompt_candidates = f.read().splitlines()
#

# template = "According to the given description, \"<desc>,\" please provide the ID of the object that closely matches this description."
# import tqdm
# for i in tqdm.tqdm(range(len(annos))):
#     obj_id = annos[i]["obj_id"]
#     if split == "train":
#         annos[i]["caption"] = annos[i]["caption"].strip()
#         if annos[i]["caption"][-1] == ".":
#             annos[i]["caption"] = annos[i]["caption"][:-1]
#         annos[i]["prompt"] = template.replace("<desc>", annos[i]["caption"])
#         annos[i]["caption"] = f"obj{obj_id:02}.".capitalize()
#     else:
#         annos[i]["ref_captions"][0] = annos[i]["ref_captions"][0].strip()
#         if annos[i]["ref_captions"][0][-1] == ".":
#             annos[i]["ref_captions"][0] = annos[i]["ref_captions"][0][:-1]
#         annos[i]["prompt"] = template.replace("<desc>", annos[i]["ref_captions"][0])
#         # assert len(annos[i]["ref_captions"]) == 1
#         annos[i]["ref_captions"] = [
#             f"obj{obj_id:02}.".capitalize()
#         ]

# with open(f"annotations/{dataset}_{split}_stage2_grounding_new.json", "w") as f:
#     json.dump(annos, f, indent=4)



template = "According to the given description, \"<desc>,\" please provide the ID of the object that closely matches this description."
new_annos = []
import tqdm
for anno in tqdm.tqdm(annos):
    scene_id = anno["scene_id"]
    obj_id = anno["obj_id"]
    if split == "train":
        caption = anno["caption"].strip()
        if caption[-1] == ".":
            caption = caption[:-1]
        prompt = template.replace("<desc>", caption)
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "prompt": prompt,
            "caption": f"<OBJ{obj_id:02}>."
        })
    else:
        for caption in anno["ref_captions"]:
            caption = caption.strip()
            if caption[-1] == ".":
                caption = caption[:-1]
            prompt = template.replace("<desc>", caption)
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "prompt": prompt,
                "ref_captions": [
                    f"<OBJ{obj_id:03}>."
                ]
            })
print(len(new_annos))

with open(f"annotations/{dataset}_{split}_stage2_grounding_OBJ.json", "w") as f:
    json.dump(new_annos, f, indent=4)


# anno_file = "/root/scene-LLaMA/datasets/referit3d/annotations/bert_tokenized/sr3d.jsonl"
# val_split_path = "/root/scene-LLaMA/datasets/referit3d/annotations/splits/scannetv2_val.txt"
# train_split_path = "/root/scene-LLaMA/datasets/referit3d/annotations/splits/scannetv2_train.txt"
# train_scene_ids = []
# val_scene_ids = []

# with open(train_split_path, "r") as f:
#     for line in f.readlines():
#         train_scene_ids.append(line.strip())

# with open(val_split_path, "r") as f:
#     for line in f.readlines():
#         val_scene_ids.append(line.strip())

# annos = []
# with jsonlines.open(anno_file, "r") as reader:
#     for l in reader:
#         annos.append(l)

# train_annos = []
# val_annos = []

# for anno in annos:
#     scene_id = anno["scan_id"]
#     obj_id = anno["target_id"]
#     caption = anno["utterance"]
#     sentences = sent_tokenize(caption)
#     altered_sentences = [' '.join(sentence.split()) for sentence in sentences]
#     caption = ' '.join([sentence.capitalize() for sentence in altered_sentences])
#     if caption[-1] == ".":
#         caption = caption[:-1]
#     prompt = template.replace("<desc>", caption)
#     if scene_id in train_scene_ids:
#         train_annos.append({
#             "scene_id": scene_id,
#             "obj_id": obj_id,
#             "prompt": prompt,
#             "caption": f"Obj{obj_id:02}."
#         })
#     else:
#         val_annos.append({
#             "scene_id": scene_id,
#             "obj_id": obj_id,
#             "prompt": prompt,
#             "ref_captions": [f"Obj{obj_id:02}."]
#         })
#
# print(len(train_annos), len(val_annos))
#
# with open("annotations/sr3d_train_stage2_grounding_new.json", "w") as f:
#     json.dump(train_annos, f, indent=4)
#
# with open("annotations/sr3d_val_stage2_grounding_new.json", "w") as f:
#     json.dump(val_annos, f, indent=4)

# with open("prompts/concise_description_objxx.txt", 'r') as f:
#     prompt_candidates = f.read().splitlines()

# for anno in annos:
#     scene_id = anno["scan_id"]
#     obj_id = anno["target_id"]
#     caption = anno["utterance"]
#     sentences = sent_tokenize(caption)
#     altered_sentences = [' '.join(sentence.split()) for sentence in sentences]
#     caption = ' '.join([sentence.capitalize() for sentence in altered_sentences])
#     # if caption[-1] == ".":
#     #     caption = caption[:-1]
#     if caption[-1] != ".":
#         caption += "."
#     prompt = random.choice(prompt_candidates).replace("<id>", f"{obj_id:02}")
#     if scene_id in train_scene_ids:
#         train_annos.append({
#             "scene_id": scene_id,
#             "obj_id": obj_id,
#             "prompt": prompt,
#             "caption": caption
#         })
#     else:
#         val_annos.append({
#             "scene_id": scene_id,
#             "obj_id": obj_id,
#             "prompt": prompt,
#             "ref_captions": [caption]
#         })

# print(len(train_annos), len(val_annos))

# with open("annotations/sr3d_train_stage2_objxx.json", "w") as f:
#     json.dump(train_annos, f, indent=4)

# with open("annotations/sr3d_val_stage2_objxx.json", "w") as f:
#     json.dump(val_annos, f, indent=4)