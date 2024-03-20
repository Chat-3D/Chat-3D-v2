import json
import os

split = 'train'
ori_anno = json.load(open(os.path.join('annotations/multi3drefer_train_val', f"multi3drefer_{split}.json")))

new_anno = []