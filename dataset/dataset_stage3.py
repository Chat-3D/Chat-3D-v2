import logging
import os
import json
import sqlite3
import random
from os.path import basename

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data
import glob

logger = logging.getLogger(__name__)


class S3PTDataset(PTBaseDataset):

    def __init__(self, ann_file, system_path="", **kwargs):
        super().__init__()
        self.feat_file, self.attribute_file, self.conv_file, repeat = ann_file[:4]
        with open(system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        self.role = ("Human", "Assistant")
        # self.pc_token = "<Target><TargetHere></Target>"
        # self.scene_token = "<Scene><SceneHere></Scene>"
        self.begin_signal = "###"
        self.end_signal = " "

        self.feats = torch.load(self.feat_file, map_location='cpu')
        self.attributes = torch.load(self.attribute_file, map_location='cpu')
        self.convs = json.load(open(self.conv_file, 'r'))
        if isinstance(self.convs, list):
            self.anno = self.convs * repeat
        else:
            annos = []
            for k, v in self.convs.items():
                if len(v) == 0:
                    continue
                tmp = k.split("_")
                obj_id = int(tmp[-1])
                scene_id = "_".join(tmp[:-1])
                annos.append({
                    "scene_id": scene_id,
                    "obj_id": obj_id,
                    "QA": v
                })
            self.anno = annos * repeat

    def __len__(self):
        return len(self.anno)

    def process_qa(self, qas, msg=""):
        conversation = self.system + self.end_signal
        for idx, qa in enumerate(qas):
            q = qa["Question"]
            a = qa["Answer"]
            conversation += (self.begin_signal + self.role[0] + ": " + q.rstrip() + self.end_signal)
            conversation += (self.begin_signal + self.role[1] + ": " + a.rstrip() + self.end_signal)
        conversation += self.begin_signal
        return conversation

    def __getitem__(self, index):
        scene_id, obj_id, target_id, scene_feat, scene_locs, scene_colors = self.get_anno(index)
        conversation = self.process_qa(self.anno[index]["QA"])
        return scene_feat, scene_locs, scene_colors, obj_id, target_id, conversation


def s3_collate_fn(batch):
    scene_feats, scene_locs, scene_colors, obj_ids, target_ids, conversations = zip(*batch)
    batch_scene_feat, batch_scene_locs, batch_scene_colors, batch_scene_mask = process_batch_data(scene_feats, scene_locs, scene_colors)
    target_ids = torch.tensor(target_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_locs": batch_scene_locs,
        "scene_colors": batch_scene_colors,
        "scene_mask": batch_scene_mask,
        "obj_id": obj_ids,
        "target_id": target_ids,
        "conversations": conversations
        # "ids": index
    }
