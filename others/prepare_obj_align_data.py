import json
import random

split = "val"
with open(f"annotations/scanrefer_{split}_stage1.json", "r") as f:
    annos = json.load(f)

if split == "train":
    with open(f"annotations/scannet_{split}_stage1.json", "r") as f:
        annos.extend(json.load(f))

print(len(annos))

new_annos = []
unwanted_words = ["wall", "ceiling", "floor", "object", "item"]

with open("prompts/obj_align_template.txt", 'r') as f:
    answer_templates = f.read().splitlines()

for anno in annos:
    scene_id = anno["scene_id"]
    obj_id = anno["obj_id"]
    caption = anno["captions"][0]
    prompt = f"What is the <OBJ{int(obj_id):03}>?"
    if any(x in caption for x in unwanted_words):
        continue
    if split == "train":
        answer_template = random.choice(answer_templates)
        if answer_template.count("{}") == 2:
            answer = answer_template.format(f"<OBJ{int(obj_id):03}>", caption)
        else:
            answer = answer_template.format(caption)
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "prompt": prompt,
            "caption": answer
        })
    else:
        answers = []
        for answer_template in answer_templates:
            if answer_template.count("{}") == 2:
                answer = answer_template.format(f"<OBJ{int(obj_id):03}>", caption)
            else:
                answer = answer_template.format(caption)
            answers.append(answer)
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "prompt": prompt,
            "ref_captions": answers
        })

print(len(new_annos))

with open(f"annotations/obj_align_{split}_OBJ.json", "w") as f:
    json.dump(new_annos, f, indent=4)
