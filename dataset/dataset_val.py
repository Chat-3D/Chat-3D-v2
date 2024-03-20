import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data, replace_old_id
import glob

logger = logging.getLogger(__name__)


class ValPTDataset(PTBaseDataset):

    def __init__(self, ann_file, system_path="", stage=2, **kwargs):
        super().__init__()
        self.feat_file, self.attribute_file, self.prompt_file = ann_file[:3]
        with open(system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        self.feats = torch.load(self.feat_file, map_location='cpu')
        self.attributes = torch.load(self.attribute_file, map_location='cpu')
        self.anno = json.load(open(self.prompt_file, 'r'))
        if stage == 2:
            self.prompt_template = "\n### Human: {}\n### Assistant:"
        else:
            self.prompt_template = "\n### Human: {}\n###"

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, obj_id, scene_feat, scene_locs, scene_colors = self.get_anno(index)
        prompt = self.system + self.prompt_template.format(self.anno[index]["prompt"])
        ref_captions = self.anno[index]["ref_captions"].copy() if "ref_captions" in self.anno[index] else []
        qid = self.anno[index]["qid"] if "qid" in self.anno[index] else 0
        obj_num = scene_locs.shape[0]
        # if obj_num > 20:
        #     pos = scene_locs[:, :3]
        #     dist = torch.sqrt(torch.sum((pos.unsqueeze(1) - pos.unsqueeze(0)) ** 2, -1) + 1e-10)
        #     valid_mask = torch.zeros(obj_num, dtype=torch.bool)
        #     for i in range(obj_num):
        #         if f"obj{i:02}" in prompt or f"Obj{i:02}" in prompt:
        #             valid_mask[i] = 1
        #             continue
        #         for caption in ref_captions:
        #             if f"obj{i:02}" in caption or f"Obj{i:02}" in caption:
        #                 valid_mask[i] = 1
        #                 break
        #     if valid_mask.sum() > 0:
        #         min_dist = dist[valid_mask].min(dim=0)[0]
        #         # foreground_mask = torch.ones(obj_num, dtype=torch.bool)
        #         # object_labels = self.attributes[scene_id]["objects"]
        #         # for i in range(obj_num):
        #         #     if object_labels[i] in ["wall", "floor", "ceiling"]:
        #         #         foreground_mask[i] = 0
        #         norm_dist = min_dist / (min_dist.max() + 1.)
        #         valid_mask[norm_dist.topk(k=20, largest=False)[1]] = 1
        #         # valid_mask = ((norm_dist < norm_dist.median()) & foreground_mask) | valid_mask
        #         dist_prob = norm_dist.masked_fill(valid_mask, 0.)
        #         final_mask = 1 - torch.bernoulli(dist_prob)
        #         prefix_sum = final_mask.cumsum(dim=0)
        #         prompt = replace_old_id(prompt, prefix_sum)
        #         for j in range(len(ref_captions)):
        #             ref_captions[j] = replace_old_id(ref_captions[j], prefix_sum)
        #         obj_id = int(prefix_sum[obj_id]) - 1
        #         scene_feat = scene_feat[final_mask.bool()]
        #         scene_locs = scene_locs[final_mask.bool()]
        #         scene_colors = scene_colors[final_mask.bool()]
        return scene_feat, scene_locs, scene_colors, obj_id, prompt, ref_captions, scene_id, qid


def valuate_collate_fn(batch):
    scene_feats, scene_locs, scene_colors, obj_ids, prompts, ref_captions, scene_ids, qids = zip(*batch)
    batch_scene_feat, batch_scene_locs, batch_scene_colors, batch_scene_mask = process_batch_data(scene_feats, scene_locs, scene_colors)
    obj_ids = torch.tensor(obj_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_locs": batch_scene_locs,
        "scene_colors": batch_scene_colors,
        "scene_mask": batch_scene_mask,
        "obj_id": obj_ids,
        "custom_prompt": prompts,
        "ref_captions": ref_captions,
        "scene_id": scene_ids,
        "qid": qids
        # "ids": index
    }

