import json
import random

split = 'val'
# ori_annos = json.load(open(f'annotations/scanrefer_{split}_stage2_objxx.json'))

# templates = [line.rstrip() for line in open('prompts/scanrefer_caption_templates.txt')]

# new_annos = []
# for anno in ori_annos:
#     obj_id = anno['obj_id']
#     anno['prompt'] = random.choice(templates).replace('<id>', f"<OBJ{obj_id:03}>")
#     new_annos.append(anno)

# with open(f'annotations/scanrefer_{split}_stage2_caption_OBJ.json', 'w') as f:
#     json.dump(new_annos, f, indent=4)

# ori_annos = json.load(open(f'annotations/nr3d_{split}_stage2_objxx.json'))

# templates = [line.rstrip() for line in open('prompts/nr3d_caption_templates.txt')]

# new_annos = []
# for anno in ori_annos:
#     obj_id = anno['obj_id']
#     anno['prompt'] = random.choice(templates).replace('<id>', f"<OBJ{obj_id:03}>")
#     new_annos.append(anno)

# with open(f'annotations/nr3d_{split}_stage2_caption_OBJ.json', 'w') as f:
#     json.dump(new_annos, f, indent=4)


for dataset in ["scanrefer", "nr3d"]:
    x = json.load(open(f'annotations/{dataset}_val_stage2_caption_OBJ.json'))
    x = random.sample(x, 100)
    with open(f'annotations/{dataset}_val_stage2_caption100_OBJ.json', 'w') as f:
        json.dump(x, f, indent=4)

# x = json.load(open('annotations/scanqa_val_stage2_objxx.json'))

# x = random.sample(x, 100)

# with open('annotations/scanqa_val_stage2_objxx100.json', 'w') as f:
#     json.dump(x, f, indent=4)


# split = 'val'
# iou = '25'

# ori_annos = json.load(open(f'annotations/scanrefer_pointgroup_{split}_stage2_caption_iou{iou}.json'))

# templates = [line.rstrip() for line in open('prompts/scanrefer_caption_templates.txt')]

# new_annos = []
# for anno in ori_annos:
#     obj_id = anno['obj_id']
#     anno['prompt'] = random.choice(templates).replace('<id>', f"{obj_id:02}")
#     new_annos.append(anno)

# with open(f'annotations/scanrefer_pointgroup_{split}_stage2_caption_iou{iou}.json', 'w') as f:
#     json.dump(new_annos, f, indent=4)