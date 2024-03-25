import torch
import json

# annos = json.load(open("annotations/obj_align_train_OBJ.json"))

# new_annos = []
# obj_ids = set()

# for anno in annos:
#     if anno['scene_id'] == "scene0000_00":
#         tmp_id = int(anno["caption"].split("OBJ")[1][:3])
#         if tmp_id in obj_ids:
#             continue
#         obj_ids.add(tmp_id)
#         if anno["caption"].startswith("<OBJ"):
#             anno["caption"] = "The " + anno["caption"]
        
#         anno["ref_captions"] = [anno["caption"]]
#         del anno["caption"]
#         new_annos.append(anno)

# print(len(new_annos))
# with open("annotations/obj_align_val_one_scene.json", "w") as f:
#     json.dump(new_annos, f, indent=4)

annos = json.load(open("/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/annotations/scanrefer_mask3d_val_stage2_grounding_OBJ.json"))

import random
annos = random.sample(annos, 200)

with open("annotations/scanrefer_mask3d_val_stage2_grounding_OBJ100.json", "w") as f:
    json.dump(annos, f, indent=4)
