import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data, replace_old_id
import glob

logger = logging.getLogger(__name__)



class S2PTDataset(PTBaseDataset):

    def __init__(self, ann_file, **kwargs):
        super().__init__()
        self.feat_file, self.attribute_file, self.anno_file = ann_file[:3]

        self.feats = torch.load(self.feat_file, map_location='cpu')
        self.attributes = torch.load(self.attribute_file, map_location='cpu')
        self.anno = json.load(open(self.anno_file, 'r'))

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, obj_id, scene_feat, scene_locs, scene_colors = self.get_anno(index)
        caption = self.anno[index]["caption"]
        question = self.anno[index]["prompt"]
        related_ids = self.anno[index]["related_ids"] if "related_ids" in self.anno[index] else None
        obj_num = scene_locs.shape[0]
        # if obj_num > 20:
        #     pos = scene_locs[:, :3]
        #     dist = torch.sqrt(torch.sum((pos.unsqueeze(1) - pos.unsqueeze(0)) ** 2, -1) + 1e-10)
        #     valid_mask = torch.zeros(obj_num, dtype=torch.bool)
        #     for i in range(obj_num):
        #         if f"obj{i:02}" in caption or f"obj{i:02}" in question or \
        #                 f"Obj{i:02}" in caption or f"Obj{i:02}" in question:
        #             valid_mask[i] = 1
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
        #         caption = replace_old_id(caption, prefix_sum)
        #         question = replace_old_id(question, prefix_sum)
        #         obj_id = int(prefix_sum[obj_id]) - 1
        #         scene_feat = scene_feat[final_mask.bool()]
        #         scene_locs = scene_locs[final_mask.bool()]
        #         scene_colors = scene_colors[final_mask.bool()]
        #         if related_ids is not None:
        #             related_ids = [int(prefix_sum[x])-1 for x in related_ids]
        detach_mask = torch.ones(scene_feat.shape[0], dtype=torch.bool)
        if related_ids is not None:
            for rid in related_ids:
                detach_mask[rid] = 0
        return scene_feat, scene_locs, scene_colors, obj_id, caption, question, detach_mask


def s2_collate_fn(batch):
    scene_feats, scene_locs, scene_colors, obj_ids, captions, questions, detach_masks = zip(*batch)
    batch_scene_feat, batch_scene_locs, batch_scene_colors, batch_scene_mask = process_batch_data(scene_feats,
                                                                                                  scene_locs,
                                                                                                  scene_colors)
    batch_detach_mask = torch.ones_like(batch_scene_mask, dtype=torch.bool)
    for i in range(batch_detach_mask.shape[0]):
        batch_detach_mask[i][:detach_masks[i].shape[0]] = detach_masks[i]
    obj_ids = torch.tensor(obj_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_locs": batch_scene_locs,
        "scene_colors": batch_scene_colors,
        "scene_mask": batch_scene_mask,
        "detach_mask": batch_detach_mask,
        "obj_ids": obj_ids,
        "answers": captions,
        "questions": questions
        # "ref_captions": ref_captions,
        # "ids": index
    }
