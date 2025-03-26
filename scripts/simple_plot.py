import os
from matplotlib import pyplot as plt
import cv2
import matplotlib.patches as mpatches
import numpy as np
from utils import plot_boxes, plot_oriented_boxes_bev, plot_pred_oriented_bboxes, draw_legend

# Load the GT
script_dir = os.path.dirname(os.path.abspath(__file__))
label_dir_path = os.path.join(script_dir, '../data/labels')
label_files = os.listdir(label_dir_path)
label_files = [file for file in label_files if file.endswith('.txt')]

if label_files:
    # gt_path = os.path.join(label_dir_path, label_files[0])    # Open the first
    gt_path = os.path.join(script_dir, '../data/labels/d46ff5df-95e8-32da-a0d7-87f7b976a959_315971495049927217.txt')    # Choose the file you want to look at
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
    model_preds.append(np.atleast_2d(np.loadtxt(pred_path, delimiter=',')))

# Plot 2D Boxes
img = plot_boxes(img, gt, 'g', 'Ground Truth')
colors = ['r', 'b']
for i, model_pred in enumerate(model_preds):
    plot_boxes(img, model_pred, colors[i], model_names[i])
fig, ax = plt.subplots(1, figsize=(12, 8))

legend_entries = [("Model A", 'r'), ("Model B", 'b'), ("Ground Truth", 'g')]
img = draw_legend(img, legend_entries)

cv2.imwrite("overlapping_image.png", img)

# Plot BEV Boxes
fig, ax = plt.subplots(1, figsize=(8, 12))
ax = plot_oriented_boxes_bev(ax, gt, 'g', 'Ground Truth')
for i, model_pred in enumerate(model_preds):
    ax = plot_pred_oriented_bboxes(model_pred[:, 5:], ax, colors[i], 0.5)
ax.set_title('Bird\'s Eye View of Oriented Bounding Boxes')

legend_entries = [mpatches.Patch(color='red', label='Model A'), mpatches.Patch(color='blue', label='Model B'), mpatches.Patch(color='green', label='Ground Truth')]
ax.legend(handles=legend_entries, loc='upper right', fontsize='small', framealpha=0.8)

# Plot circles with radii 50, 100, 150
for radius in [50, 100, 150]:
    circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--')
    ax.add_patch(circle)
    
ax.set_xlim([-50, 50])
ax.set_ylim([-0, 125])
fig.canvas.draw()
plt.savefig("overlapping_bev.png")
plt.close(fig)