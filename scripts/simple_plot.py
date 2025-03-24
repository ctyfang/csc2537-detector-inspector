import os
from matplotlib import pyplot as plt
import cv2
# import csv
import numpy as np
from utils import plot_boxes, plot_oriented_boxes_bev, color_to_rgb, plot_pred_oriented_bboxes

# Load the GT
# label_dir_path = '/home/cfang/git/csc2537-detector-inspector/data/labels'
label_dir_path = 'D:/Master/Courses/Information Visualization/csc2537-detector-inspector/data/labels'
label_files = os.listdir(label_dir_path)
label_files = [file for file in label_files if file.endswith('.txt')]

if label_files:
    gt_path = os.path.join(label_dir_path, label_files[1])    # Open the first
    # gt = np.loadtxt(gt_path, delimiter=',')
    gt = np.atleast_2d(np.loadtxt(gt_path, delimiter=','))
else:
    print("No text files found in the directory.")

gt_filename = os.path.basename(gt_path)

img_path = os.path.join(label_dir_path, "..", "images", gt_filename.replace(".txt", ".jpg"))
img = cv2.imread(img_path)

# Load the model outputs
model_names = ["fcos3d", "fcos3d_finetune"]
model_preds = []
for model_name in model_names:
    pred_path = os.path.join(label_dir_path, "..", "model_outputs", model_name, gt_filename)
    assert os.path.exists(pred_path), "Corresponding predictions not found"
    # model_preds.append(np.loadtxt(pred_path, delimiter=','))
    model_preds.append(np.atleast_2d(np.loadtxt(pred_path, delimiter=',')))

    # Plot 2D Boxes
    img = plot_boxes(img, gt, 'g', 'Ground Truth')
    colors = ['r', 'b']
    for i, model_pred in enumerate(model_preds):
        plot_boxes(img, model_pred, colors[i], model_names[i])
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    ax.set_title('2D Bounding Boxes')
    plt.savefig("boxes2d_1.png")
    plt.close(fig)

    # Plot BEV Boxes
    fig, ax = plt.subplots(1, figsize=(8, 12))
    ax = plot_oriented_boxes_bev(ax, gt, 'g', 'Ground Truth')
    for i, model_pred in enumerate(model_preds):
        ax = plot_pred_oriented_bboxes(model_pred[:, 5:], ax, colors[i], 0.5)
    ax.set_title('Bird\'s Eye View of Oriented Bounding Boxes')
    
    # Plot circles with radii 50, 100, 150
    for radius in [50, 100, 150]:
        circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--')
        ax.add_patch(circle)
        
    ax.set_xlim([-50, 50])
    ax.set_ylim([-0, 125])
    fig.canvas.draw()
    plt.savefig("boxes3d_bev_1.png")
    plt.close(fig)