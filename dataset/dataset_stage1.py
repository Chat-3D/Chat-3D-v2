import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data
import glob

logger = logging.getLogger(__name__)


class S1PTDataset(PTBaseDataset):

    def __init__(self, ann_file, **kwargs):
        super().__init__()
        self.feat_file, self.attribute_file, self.anno_file = ann_file[:3]

        # self.cat2id = json.load(open("annotations/cat2nyu40id.json", "r"))

        self.feats = torch.load(self.feat_file, map_location='cpu')
        self.attributes = torch.load(self.attribute_file, map_location='cpu')
        self.anno = json.load(open(self.anno_file, 'r'))

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, obj_id, scene_feat, scene_locs, scene_colors = self.get_anno(index, stage=1)
        scene_feat, scene_locs, scene_colors = scene_feat[obj_id:obj_id+1], scene_locs[obj_id:obj_id+1], scene_colors[obj_id:obj_id+1]
        target_captions = self.anno[index]["captions"]
        assert len(target_captions) > 0 and len(target_captions[0]) > 0, target_captions
        # obj_labels = self.attributes[scene_id]["objects"]
        # target_cls = self.cat2id[obj_labels[obj_id]] - 1
        # target_cls = [self.cat2id[x] - 1 for x in obj_labels]
        return scene_feat, scene_locs, scene_colors, target_captions


def s1_collate_fn(batch):
    scene_feats, scene_locs, scene_colors, target_captions = zip(*batch)
    # batch_scene_feat, batch_scene_locs, batch_scene_colors, batch_scene_mask = process_batch_data(scene_feats, scene_locs, scene_colors)
    # batch_target_clses = torch.zeros_like(batch_scene_mask, dtype=torch.long)
    # for i in range(len(batch_target_clses.shape[0])):
    #     batch_target_clses[i][:target_clses[i].shape[0]] = target_clses
    scene_feats = torch.stack(scene_feats)
    scene_locs = torch.stack(scene_locs)
    scene_colors = torch.stack(scene_colors)
    # target_clses = torch.tensor(target_clses, dtype=torch.long)
    return {
        "scene_feat": scene_feats,
        "scene_locs": scene_locs,
        "scene_colors": scene_colors,
        # "scene_mask": batch_scene_mask,
        # "obj_ids": obj_ids,
        "target_captions": target_captions,
        # "target_clses": target_clses,
        # "ids": index
    }

