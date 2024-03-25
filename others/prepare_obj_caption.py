import json
import os
import random

split = 'train'
all_objects = json.load(open('annotations/all_objects_with_full_desc.json'))
templates = [line.rstrip() for line in open('prompts/object_caption_templates.txt')]
scan_list = [line.rstrip() for line in open(f"annotations/scannet_{split}.txt")]

new_annos = []
for k, v in all_objects.items():
    if k not in scan_list:
        continue
    for o in v['object_list']:
        if o['description'] is not None:
            new_annos.append({
                'scene_id': k,
                'obj_id': o['object_id'],
                'prompt': random.choice(templates).replace('<id>', f"<OBJ{o['object_id']:03}>"),
                'caption': o['description'],
                'related_ids': [o['object_id']]
            })
print(len(new_annos))

with open(f'annotations/scannet_{split}_stage2_caption_OBJ.json', 'w') as f:
    json.dump(new_annos, f, indent=4)
