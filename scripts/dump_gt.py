"""
This script (adapted from https://github.com/megvii-research/Far3D/blob/main/tools/visual/vis_av2.py) 
is used to dump ground truth 2D and 3D bounding boxes along with the camera calibration data.

The script performs the following steps:
1. Loads the dataset configuration and builds the dataset using the `mmdet3d` library.
2. Iterates through the dataset at specified intervals.
3. For each data point, extracts the scene ID and timestamp from the image metadata.
4. Saves the image to the specified directory.
5. Saves the camera intrinsics and extrinsics to the specified directory.
6. Extracts and processes the ground truth 2D and 3D bounding boxes.
7. Saves the processed ground truth labels to a text file.

- Processed images saved as JPEG files.
- Camera calibration data saved as PyTorch tensors.
- Ground truth labels saved as text files.
"""

from mmdet3d.datasets import build_dataset
import os
import importlib
import numpy as np
import cv2
import torch
from tqdm import tqdm
        
pkl_path = 'av2_val_infos_withmap_full.pkl'
save_pth = "/home/cfang/git/csc2537-detector-inspector/data"

if not os.path.exists(save_pth):
    os.mkdir(save_pth)
plugin_dir = 'projects/mmdet3d_plugin/'
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]

for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
print(_module_path)
plg_lib = importlib.import_module(_module_path)

collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
               'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
               'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
               'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
               'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
               'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']
point_cloud_range = [0, -152.4, -5.0, 152.4, 152.4, 5.0]

no_aug_conf = {
        "resize_lim": (0.625, 0.625),
        "final_dim": (640, 960),
        "final_dim_f": (640, 720),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "rand_flip": False,
        "test": True
    }

vis_pipeline = [
    dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectClassCameraFilter', all_classes=class_names),
    dict(type='ObjectRangeFilterFrontCam', point_cloud_range=point_cloud_range),
    dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=no_aug_conf),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d','lidar_timestamp'))
]
data_root = '/mnt/remote/shared_data/datasets/argoverse_v2_sensor_updated/'
input_image_dir = '/home/cfang/git/csc2537-detector-inspector/data/images_orig/'
vis_config = dict(
    type = 'Argoverse2DatasetT',
    data_root = data_root,
    collect_keys=collect_keys + ['img', 'img_metas'], 
    queue_length=1, 
    ann_file=data_root + pkl_path, 
    split='val',
    load_interval=1,
    classes=class_names, 
    interval_test=False,
    pipeline=vis_pipeline,
)
dataset = build_dataset(vis_config)

for data_idx in tqdm(range(0, 23000, 500)):
    data = dataset[data_idx]
    scene_id = data['img_metas'].data['scene_token']
    timestamp_ns = int(os.path.split(data['img_metas'].data['filename'][0])[1].split(".")[0])
    img = data['img'].data[0][0]
    img = np.ascontiguousarray(img.numpy().transpose(1, 2, 0))
    cv2.imwrite(os.path.join(save_pth, "images", f"{scene_id}_{timestamp_ns}.jpg"), img)
    
    torch.save(data['intrinsics'], os.path.join(save_pth, "calibration", f"{scene_id}_{timestamp_ns}_intrinsics.pt"))
    torch.save(data['extrinsics'], os.path.join(save_pth, "calibration", f"{scene_id}_{timestamp_ns}_extrinsics.pt"))

    gt_bboxes_2d = data['gt_bboxes'].data[0].numpy()
    gt_bboxes_3d = data['gt_bboxes_3d'].data.corners
    gt_bboxes_3d_center = data['gt_bboxes_3d'].data.tensor[:, :3]
    gt_bboxes_3d_center = np.concatenate((gt_bboxes_3d_center[:, :3], np.ones(gt_bboxes_3d_center.shape[0])[:, None]), axis=1)
    gt_bboxes_3d_center = (data['extrinsics'].data[0][0] @ gt_bboxes_3d_center.T).T[:, :-1]
    gt_bboxes_3d_size = data['gt_bboxes_3d'].data.tensor[:, [5,4,3]]
    gt_bboxes_xyzlwhyaw = torch.cat([gt_bboxes_3d_center, gt_bboxes_3d_size, data['gt_bboxes_3d'].data.tensor[:, -1].unsqueeze(1)], dim=1).numpy()
    gt_labels = data['gt_labels'].data[0].numpy()
    
    gt_labels_all = np.concatenate([gt_labels.reshape(-1, 1), gt_bboxes_2d.reshape(-1, 4), gt_bboxes_xyzlwhyaw.reshape(-1, 7)], axis=-1)
    with open(os.path.join(save_pth, "labels", f"{scene_id}_{timestamp_ns}.txt"), "w") as out_f:
        for gt_label in gt_labels_all:
            out_str = ",".join([str(x) for x in gt_label.tolist()]) + "\n"
            out_f.write(out_str)