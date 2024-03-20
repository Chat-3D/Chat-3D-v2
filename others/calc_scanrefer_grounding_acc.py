import numpy as np
import json
import sys
import torch
import random
from tqdm import tqdm

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


output_file = "/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240307_190254_dp0.1_lr5e-4_sta2_ep3_bs3*4_grounding_onlyscanrefer/preds_epoch2_step15072.json"
outputs = json.load(open(output_file, "r"))


instance_attribute_file = "annotations/scannet_mask3d_val_attributes.pt"
scannet_attribute_file = "annotations/scannet_val_attributes.pt"

instance_attrs = torch.load(instance_attribute_file)
scannet_attrs = torch.load(scannet_attribute_file)

iou25_acc = 0
iou50_acc = 0

count_list = [0] * 150
iou25_acc_list = [0] * 150
iou50_acc_list = [0] * 150

for i, output in tqdm(enumerate(outputs)):
    scene_id = output["scene_id"]
    obj_id = output["obj_id"]
    instance_locs = instance_attrs[scene_id]["locs"]
    scannet_locs = scannet_attrs[scene_id]["locs"]
    pred = output["pred"]
    instance_num = instance_locs.shape[0]
    pred_id = random.randint(0, instance_num-1)
    pred = pred.capitalize()
    pred = pred.replace("object", "")
    pred = pred.replace("Object", "")
    # print(pred)
    # breakpoint()
    flag = 0
    if "Obj" in pred:
        tmp = pred.split("Obj")[1]
        j = 0
        while tmp[:j+1].isdigit() and j < len(tmp):
            j = j + 1
        if j > 0:
            flag = 1
            if int(tmp[:j]) < instance_num:
                pred_id = int(tmp[:j])
    if not flag and "obj" in pred:
        tmp = pred.split("obj")[1]
        j = 0
        while tmp[:j+1].isdigit() and j < len(tmp):
            j = j + 1
        if j > 0:
            flag = 1
            if int(tmp[:j]) < instance_num:
                pred_id = int(tmp[:j])
    # print(pred_id)
    pred_locs = instance_locs[pred_id].tolist()
    gt_locs = scannet_locs[obj_id].tolist()
    pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
    gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
    iou = box3d_iou(pred_corners, gt_corners)
    if iou >= 0.25:
        iou25_acc += 1
        iou25_acc_list[scannet_locs.shape[0]] += 1
    if iou >= 0.5:
        iou50_acc += 1
        iou50_acc_list[scannet_locs.shape[0]] += 1
    count_list[scannet_locs.shape[0]] += 1
    # max_iou = 0.
    # for pred_id in range(instance_num):
    #     pred_locs = instance_locs[pred_id].tolist()
    #     gt_locs = scannet_locs[obj_id].tolist()
    #     pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
    #     gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
    #     iou = box3d_iou(pred_corners, gt_corners)
    #     if iou > max_iou:
    #         max_iou = iou
    # if max_iou >= 0.25:
    #     iou25_acc += 1
    # if max_iou >= 0.5:
    #     iou50_acc += 1
    # print(iou)

print(f"Acc 0.25 {float(iou25_acc) / len(outputs)}")
print(f"Acc 0.50 {float(iou50_acc) / len(outputs)}")


split_nums = [0, 25, 35, 50, 70, 150]

for i in range(len(split_nums)-1):
    tot, iou25, iou50 = 0, 0, 0
    for j in range(split_nums[i], split_nums[i+1]):
        tot += count_list[j]
        iou25 += iou25_acc_list[j]
        iou50 += iou50_acc_list[j]
    print(f"{split_nums[i]} <= x < {split_nums[i+1]}: {tot} {iou25 / tot} {iou50 / tot}")

# for i in range(len(iou25_acc_list)):
#     iou25_acc_list[i] = iou25_acc_list[i] / count_list[i] if count_list[i] > 0 else 0
# for i in range(len(iou50_acc_list)):
#     iou50_acc_list[i] = iou50_acc_list[i] / count_list[i] if count_list[i] > 0 else 0



# print(iou25_acc_list)
# print(iou50_acc_list)