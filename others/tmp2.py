import json
import numpy as np

split = 'val'

scanqa_anno = json.load(open(f"annotations/scanqa_{split}_stage2_objxx.json"))
scanrefer_anno = json.load(open(f"annotations/scanrefer_{split}_stage2_objxx.json"))

scanqa_scan_list = np.unique([x['scene_id'] for x in scanqa_anno])
scanrefer_scan_list = np.unique([x['scene_id'] for x in scanrefer_anno])

print(len(set(scanrefer_scan_list) - set(scanqa_scan_list)))
print(len(scanqa_scan_list))
print(len(scanrefer_scan_list))
