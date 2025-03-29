import os
from matplotlib import pyplot as plt
import cv2
import matplotlib.patches as mpatches
import numpy as np
from mmcv.ops import box_iou_rotated
import torch

import debugpy
import os
if os.environ.get("ENABLE_DEBUGPY"):
    print("listening...")
    debugpy.listen(("127.0.0.1", 5678))
    debugpy.wait_for_client()
        
# Load the GT
script_dir = os.path.dirname(os.path.abspath(__file__))
label_dir_path = os.path.join(script_dir, '../data/labels')
label_files = os.listdir(label_dir_path)
label_files = [file for file in label_files if file.endswith('.txt')]

if label_files:
    # gt_path = os.path.join(label_dir_path, label_files[0])    # Open the first
    gt_path = os.path.join(script_dir, '../data/labels/87ca3d9f-f317-3efb-b1cb-aaaf525227e5_315969175949927216.txt')    # Choose the file you want to look at
    gt = np.atleast_2d(np.loadtxt(gt_path, delimiter=','))
else:
    print("No text files found in the directory.")

gt_filename = os.path.basename(gt_path)

# Load the model outputs
iou_thresh = 0.3
model_names = ["fcos3d", "fcos3d_finetune"]
model_preds = []
for model_name in model_names:
    pred_path = os.path.join(label_dir_path, "..", "model_outputs", model_name, gt_filename)
    assert os.path.exists(pred_path), "Corresponding predictions not found"
    model_preds.append(np.atleast_2d(np.loadtxt(pred_path, delimiter=',')))

model_metrics_bev = {n: {"tp": 0, "fp": 0, "fn": 0} for n in model_names}
model_metrics_2d = {n: {"tp": 0, "fp": 0, "fn": 0} for n in model_names}
for model_name, model_pred in zip(model_names, model_preds):
    ious = box_iou_rotated(torch.from_numpy(model_pred[:, [5, 7, 8, 9]]).float(), 
                    torch.from_numpy(gt[:, [5, 7, 8, 9]]).float())
    ious_match, gt_idcs_match = ious.max(dim=-1)
    tp = sum(ious_match >= iou_thresh)
    fp = sum(ious_match < iou_thresh)
    fn = torch.clamp(gt.shape[0] - tp, min=0).item()
    model_metrics_bev[model_name]["tp"] += tp
    model_metrics_bev[model_name]["fp"] += fp
    model_metrics_bev[model_name]["fn"] += fn
    
    xyxy_pred = torch.from_numpy(model_pred[:, [1, 2, 3, 4]]).float()
    xyxy_gt = torch.from_numpy(gt[:, [1, 2, 3, 4]]).float()
    
    # Convert boxes from xyxy to xywh and concatenate zero yaw
    xywh_pred = torch.zeros((xyxy_pred.shape[0], 5), dtype=xyxy_pred.dtype, device=xyxy_pred.device)
    xywh_pred[:, 0] = (xyxy_pred[:, 0] + xyxy_pred[:, 2]) / 2  # x center
    xywh_pred[:, 1] = (xyxy_pred[:, 1] + xyxy_pred[:, 3]) / 2  # y center
    xywh_pred[:, 2] = xyxy_pred[:, 2] - xyxy_pred[:, 0]        # width
    xywh_pred[:, 3] = xyxy_pred[:, 3] - xyxy_pred[:, 1]        # height
    xywh_pred[:, 4] = 0                                        # yaw (zero)

    xywh_gt = torch.zeros((xyxy_gt.shape[0], 5), dtype=xyxy_gt.dtype, device=xyxy_gt.device)
    xywh_gt[:, 0] = (xyxy_gt[:, 0] + xyxy_gt[:, 2]) / 2        # x center
    xywh_gt[:, 1] = (xyxy_gt[:, 1] + xyxy_gt[:, 3]) / 2        # y center
    xywh_gt[:, 2] = xyxy_gt[:, 2] - xyxy_gt[:, 0]              # width
    xywh_gt[:, 3] = xyxy_gt[:, 3] - xyxy_gt[:, 1]              # height
    xywh_gt[:, 4] = 0                                          # yaw (zero)

    # Use xywh in the box IoU calculation
    ious_2d = box_iou_rotated(xywh_pred, xywh_gt)
    ious_match_2d, gt_idcs_match_2d = ious_2d.max(dim=-1)
    tp_2d = sum(ious_match_2d >= iou_thresh)
    fp_2d = sum(ious_match_2d < iou_thresh)
    fn_2d = torch.clamp(gt.shape[0] - tp_2d, min=0).item()
    model_metrics_2d[model_name]["tp"] += tp_2d
    model_metrics_2d[model_name]["fp"] += fp_2d
    model_metrics_2d[model_name]["fn"] += fn_2d

for model_name in model_names:
    tp = model_metrics_bev[model_name]["tp"]
    fp = model_metrics_bev[model_name]["fp"]
    fn = model_metrics_bev[model_name]["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Model: {model_name}")
    print(f"Precision BEV: {precision:.4f}")
    print(f"Recall BEV: {recall:.4f}")
    print("-" * 30)
    
    tp_2d = model_metrics_2d[model_name]["tp"]
    fp_2d = model_metrics_2d[model_name]["fp"]
    fn_2d = model_metrics_2d[model_name]["fn"]
    precision_2d = tp_2d / (tp_2d + fp_2d) if (tp_2d + fp_2d) > 0 else 0
    recall_2d = tp_2d / (tp_2d + fn_2d) if (tp_2d + fn_2d) > 0 else 0
    print(f"Precision 2D: {precision_2d:.4f}")
    print(f"Recall 2D: {recall_2d:.4f}")
    print("=" * 30)