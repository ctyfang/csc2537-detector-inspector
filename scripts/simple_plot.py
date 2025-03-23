import os
from matplotlib import pyplot as plt
import cv2
import csv
import numpy as np


# Helper functions
def color_to_rgb(color):
    color_dict = {
        'r': (255, 0, 0),
        'g': (0, 255, 0),
        'b': (0, 0, 255)
    }
    return color_dict.get(color, (255, 255, 255))  # Default to white if color not found

# Function to plot 2D bounding boxes
def plot_boxes(img, boxes, color, label):
    for box in boxes:
        class_id, x1, y1, x2, y2 = box[:5]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_to_rgb(color), 2)
    return img
    
# Plot 3D Boxes
def plot_oriented_boxes_bev(ax, boxes, color, label):
    for box in boxes:
        x, y, z, l, w, h, yaw = box[5:]
        corners = np.array([
            [w / 2, l / 2],
            [-w / 2, l / 2],
            [-w / 2, -l / 2],
            [w / 2, -l / 2]
        ])
        
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + np.array([x, z])
        polygon = plt.Polygon(translated_corners, edgecolor=color, fill=False, linewidth=2, label=label)
        ax.add_patch(polygon)
    return ax

def plot_pred_oriented_bboxes(boxes, ax=None, color=None, alpha=0.7, linewidth=2):
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
    alpha : float, optional
        Transparency level of the bounding boxes.
    linewidth : float, optional
        Width of the bounding box edges.

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
        polygon = plt.Polygon(bottom_corners, closed=True, 
                        fill=False, alpha=alpha, 
                        edgecolor=color, facecolor=color,
                        linewidth=linewidth)
        ax.add_patch(polygon)
    return ax

# Load the GT
label_dir_path = '/home/cfang/git/csc2537-detector-inspector/data/labels'
label_files = os.listdir(label_dir_path)
label_files = [file for file in label_files if file.endswith('.txt')]

if label_files:
    gt_path = os.path.join(label_dir_path, label_files[0])    # Open the first
    gt = np.loadtxt(gt_path, delimiter=',')
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
    model_preds.append(np.loadtxt(pred_path, delimiter=','))

    # Plot 2D Boxes
    img = plot_boxes(img, gt, 'g', 'Ground Truth')
    colors = ['r', 'b']
    for i, model_pred in enumerate(model_preds):
        plot_boxes(img, model_pred, colors[i], model_names[i])
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    ax.set_title('2D Bounding Boxes')
    plt.savefig("boxes2d.png")
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
    plt.savefig("boxes3d_bev.png")
    plt.close(fig)