import torch

# a = torch.load('annotations/scannet_train_attributes.pt')
a = torch.load('annotations/scannet_mask3d_val_attributes.pt')

tot = 0
for k, v in a.items():
    tot += len(v['locs'])

print(tot / len(a))

# import json

# x = json.load(open('annotations/scanrefer_mask3d_train_stage2_grounding_new.json'))
# print(len(x))