# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

from tqdm import tqdm
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmcv.ops import nms_rotated

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
import cv2
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis
from torch import Tensor

def get_corners(box_tensor, box_dims_tensor) -> Tensor:
    """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
    x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

    .. code-block:: none

                                        up z
                        front x           ^
                                /            |
                            /             |
                (x1, y0, z1) + -----------  + (x1, y1, z1)
                            /|            / |
                            / |           /  |
            (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y <------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Returns:
        Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
    """
    if box_tensor.numel() == 0:
        return torch.empty([0, 8, 3], device=box_tensor.device)

    dims = box_dims_tensor
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
            device=dims.device, dtype=dims.dtype)
    dims = dims[..., [0, 1, 2]]
    
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin (0.5, 0.5, 0.)
    corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.5])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    corners = rotation_3d_in_axis(
        corners, box_tensor[:, 6], axis=1)
    corners += box_tensor[:, [0, 1, 2]].view(-1, 1, 3)
    return corners
    
def plot_oriented_bboxes(boxes, ax=None, colors=None, labels=None, alpha=0.7, linewidth=2, 
                         height_color_scale=False):
    """
    Plot oriented bounding boxes in bird's eye view. Assuming the 3D boxes are in a standard camera frame.
    
    Parameters:
    -----------
    boxes : List of numpy arrays or List of lists
        Each box is represented as [x, y, z, l, w, h, yaw] where:
        - x, y, z: Center coordinates
        - l, w, h: Length, width, height
        - yaw: Rotation angle in radians (0 = x-axis, counterclockwise positive)
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, a new figure and axes will be created.
    colors : List of colors or string, optional
        Colors for each box. If None, random colors will be assigned.
    labels : List of strings, optional
        Labels for each box to be shown in the legend.
    alpha : float, optional
        Transparency level of the bounding boxes.
    linewidth : float, optional
        Width of the bounding box edges.
    plot_3d : bool, optional
        If True, plot in 3D. Otherwise, plot in 2D (bird's eye view).
    height_color_scale : bool, optional
        If True, use height (z) to scale the color intensity of boxes.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    if boxes is None:
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
        ax.set_aspect('equal')
        ax.grid(True)
        return fig, ax
        
    n_boxes = len(boxes)
    
    # Prepare colors
    if colors is None:
        # Generate distinct colors
        color_list = list(mcolors.TABLEAU_COLORS.values())
        # If we need more colors than available in the predefined list
        if n_boxes > len(color_list):
            color_list = [np.random.rand(3) for _ in range(n_boxes)]
        colors = [color_list[i % len(color_list)] for i in range(n_boxes)]
    elif isinstance(colors, str):
        colors = [colors] * n_boxes
    
    # Collect all points for setting axis limits later
    all_corners = []
    
    # Process and plot each box
    for i, box in enumerate(boxes):
        # Ensure box is a numpy array
        box = np.array(box)
        
        # Extract parameters
        x, y, z, l, w, h, yaw = box
        yaw = -yaw
        
        # Create corner points of the box (in vehicle coordinates)
        # Order: Front-right, front-left, rear-left, rear-right
        corners_local = np.array([
            [l/2, w/2, 0],  # Front-right bottom
            [l/2, -w/2, 0],  # Front-left bottom
            [-l/2, -w/2, 0],  # Rear-left bottom
            [-l/2, w/2, 0],   # Rear-right bottom
        ])
        
        # Rotation matrix for yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        
        # Rotate corners and translate to world coordinates
        corners_world = []
        for corner in corners_local:
            rotated = R @ corner
            translated = rotated + np.array([x, z, 0])
            corners_world.append(translated)
            all_corners.append(translated)
        
        corners_world = np.array(corners_world)
    
        # For 2D plotting, use only the bottom corners
        bottom_corners = corners_world[:4, :2]  # Only x, z coordinates of bottom face
        
        # Create polygon
        box_color = None
        polygon = Polygon(bottom_corners, closed=True, 
                        fill=True, alpha=alpha, 
                        edgecolor=box_color, facecolor=box_color,
                        linewidth=linewidth)
        ax.add_patch(polygon)
        
        # Add center point
        ax.scatter(x, z, color=box_color, s=30, marker='x')
        
        # Add direction indicator
        # Calculate front center of the box
        front_center = np.mean(corners_world[:2, :2], axis=0)
        arrow_length = 0.8  # Scale factor for arrow length
        ax.arrow(x, z, 
                (front_center[0] - x) * arrow_length, 
                (front_center[1] - z) * arrow_length,
                head_width=0.2, head_length=0.3, 
                fc=box_color, ec=box_color)
    
    # Add legend if labels are provided
    if labels is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(min(n_boxes, len(labels)))]
        ax.legend(handles, labels[:n_boxes], loc='upper right')
    
    # Set axis limits based on all corner points
    all_corners = np.array(all_corners)
    if len(all_corners) > 0:
        x_min, z_min = np.min(all_corners[:, :2], axis=0) - 1
        x_max, z_max = np.max(all_corners[:, :2], axis=0) + 1
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
    
    # Set axis properties
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.grid(True)
    
    plt.tight_layout()
    return fig, ax

    
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load model and data config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
                
    # set multi-process settings
    setup_multi_processes(cfg)
    
    cfg.model.pretrained = None
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).eval().cuda()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    data_loader_iter = iter(data_loader)
    score_thresh = 0.15
    iou_thresh = 0.3
    save_pth = "/home/cfang/git/csc2537-detector-inspector/data/model_outputs"
    
    for data_idx in tqdm(range(len(data_loader))):
        sample = next(data_loader_iter)
        img_metas = sample['img_metas'][0].data[0][0]
        img = sample['img'][0].data[0]
        img_metas['cam2img'] = sample['intrinsics'][0].data[0][0].squeeze(0)
        img_metas['scale_factor'] = 4
        img_metas['box_type_3d'] = sample['img_metas'][0].data[0][0]['box_type_3d']
        output = model.forward_test([img.squeeze(0).cuda()], [[img_metas]])[0]['img_bbox']
        
        corners = get_corners(output['boxes_3d'].tensor, output['boxes_3d'].dims)
        centers = output['boxes_3d'].center
        dims = output['boxes_3d'].dims
        yaws = output['boxes_3d'].yaw
        labels_3d = output['labels_3d']
        scores = output['scores_3d']
        
        # NMS
        xywhyaw = torch.cat([centers[..., [0, 2]], dims[..., [1, 0]], -yaws[..., None]], dim=-1)
        _, keep_ids = nms_rotated(xywhyaw, scores, iou_threshold=iou_thresh)
        corners = corners[keep_ids]
        centers = centers[keep_ids]
        dims = dims[keep_ids]
        yaws = yaws[keep_ids]
        labels_3d = labels_3d[keep_ids]
        scores = scores[keep_ids]
        
        # Filter boxes by confidence score
        mask = scores > score_thresh
        corners = corners[mask]
        centers = centers[mask]
        dims = dims[mask]
        yaws = yaws[mask]
        labels_3d = labels_3d[mask]
        scores = scores[mask]
        
        # Construct 2D and 3D labels
        camera_matrix = img_metas['cam2img'][:3, :3]
        corners_flat = corners.view(-1, 3)
        corners_flat_uvw = (camera_matrix @ corners_flat.T).T
        corners_flat_uv = corners_flat_uvw[..., :2] / corners_flat_uvw[..., 2:3]
        corners_uv = corners_flat_uv.reshape(-1, 8, 2)
        corners_xmin = corners_uv[:, :, 0].min(dim=-1).values
        corners_xmax = corners_uv[:, :, 0].max(dim=-1).values
        corners_ymin = corners_uv[:, :, 1].min(dim=-1).values
        corners_ymax = corners_uv[:, :, 1].max(dim=-1).values
        xyxy = np.concatenate([corners_xmin[..., None], corners_ymin[..., None], corners_xmax[..., None], corners_ymax[..., None]], axis=-1)
        xyxy = xyxy.astype(np.int)
        xyzlwhyaw = torch.cat([centers, dims, yaws[..., None]], dim=-1)
        
        # Dump labels
        scene_id = sample['img_metas'][0].data[0][0]['scene_token']
        timestamp_ns = int(os.path.split(sample['img_metas'][0].data[0][0]['filename'][0])[1].split(".")[0])
        labels_all = torch.cat([labels_3d[..., None], torch.from_numpy(xyxy), xyzlwhyaw], dim=-1)
        with open(os.path.join(save_pth, "fcos3d_finetune", f"{scene_id}_{timestamp_ns}.txt"), "w") as out_f:
            for label in labels_all:
                out_str = ",".join([str(x) for x in label.tolist()]) + "\n"
                out_f.write(out_str)

        # # Debug: Visualize boxes in BEV and Image
        # img_plot = np.ascontiguousarray(img[0][0].cpu().numpy().transpose(1, 2, 0))
        # img_plot[..., 0] += 103.530
        # img_plot[..., 1] += 116.280
        # img_plot[..., 2] += 123.675
        # img_plot = img_plot.astype(np.uint8)
        
        #   # Image-View
        # for xyxy_i in xyxy:
        #     img_plot = cv2.rectangle(img_plot, (xyxy_i[0], xyxy_i[1]), (xyxy_i[2], xyxy_i[3]), (0, 100, 255), 2)
        # cv2.imwrite("hi.png", img_plot)
        
        #   # BEV        
        # xyzlwhyaw = torch.cat([centers, dims, yaws[..., None]], dim=-1)
        # plot_oriented_bboxes(xyzlwhyaw)
        # plt.savefig("bev.png")
        # plt.close()
        
        # gt_bboxes_2d = data['gt_bboxes'].data[0].numpy()
        # gt_bboxes_3d = data['gt_bboxes_3d'].data.corners
        # gt_bboxes_3d_center = data['gt_bboxes_3d'].data.tensor[:, :3]
        # gt_bboxes_3d_center = np.concatenate((gt_bboxes_3d_center[:, :3], np.ones(gt_bboxes_3d_center.shape[0])[:, None]), axis=1)
        # gt_bboxes_3d_center = (data['extrinsics'].data[0][0] @ gt_bboxes_3d_center.T).T[:, :-1]
        # gt_bboxes_3d_size = data['gt_bboxes_3d'].data.tensor[:, [5,4,3]]
        # gt_bboxes_xyzlwhyaw = torch.cat([gt_bboxes_3d_center, gt_bboxes_3d_size, data['gt_bboxes_3d'].data.tensor[:, -1].unsqueeze(1)], dim=1).numpy()
        # gt_labels = data['gt_labels'].data[0].numpy()


if __name__ == '__main__':
    import debugpy
    import os
    if os.environ.get("ENABLE_DEBUGPY"):
        print("listening...")
        debugpy.listen(("127.0.0.1", 5678))
        debugpy.wait_for_client()
    main()