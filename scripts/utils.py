import numpy as np
import cv2
import matplotlib.pyplot as plt

# Helper functions
def color_to_rgb(color):
    """
    Convert a color character to its corresponding RGB tuple.

    Parameters:
    color (str): A single character representing the color ('r' for red, 'g' for green, 'b' for blue).

    Returns:
    tuple: A tuple representing the RGB values of the color. Defaults to white (255, 255, 255) if the color is not found.
    """
    color_dict = {
        'r': (255, 0, 0),
        'g': (0, 255, 0),
        'b': (0, 0, 255)
    }
    return color_dict.get(color, (255, 255, 255))  # Default to white if color not found

def color_to_bgr(color):
    """
    For OpenCV as it uses BGR
    """
    color_dict = {
        'r': (0, 0, 255),   # Red
        'g': (0, 255, 0),   # Green
        'b': (255, 0, 0)    # Blue
    }
    return color_dict.get(color, (255, 255, 255))

# Function to plot 2D bounding boxes
def plot_boxes(img, boxes, color, label):
    """
    Draws bounding boxes on an image.

    Args:
        img (numpy.ndarray): The image on which to draw the bounding boxes.
        boxes (list of tuples): A list of bounding boxes, where each box is represented 
                                as a tuple (class_id, x1, y1, x2, y2).
        color (str): The color of the bounding box.
        label (str): The label to be displayed on the bounding box.

    Returns:
        numpy.ndarray: The image with the bounding boxes drawn on it.
    """
    boxes = np.atleast_2d(boxes)

    if isinstance(color, str):
        draw_color = color_to_bgr(color)
    elif isinstance(color, tuple) and max(color) <= 1.0:
        # If matplotlib color [0-1], convert to OpenCV BGR
        draw_color = tuple(int(c * 255) for c in color)
    else:
        draw_color = color  # Already OpenCV-compatible tuple

    for box in boxes:
        class_id, x1, y1, x2, y2 = box[:5]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), draw_color, 2)
    return img
    
# Plot 3D Boxes
def plot_oriented_boxes_bev(ax, boxes, color, label):
    """
    Plots oriented bounding boxes in bird's eye view (BEV) on a given matplotlib axis.
    Parameters:
    ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
    boxes (list of lists or numpy.ndarray): A list or array of bounding boxes, where each box is represented by 
                                            [x, y, z, l, w, h, yaw]. 
                                            - x, y, z: The center coordinates of the box.
                                            - l, w, h: The length, width, and height of the box.
                                            - yaw: The rotation angle of the box around the z-axis.
    color (str or tuple): The color of the bounding box edges.
    label (str): The label for the bounding boxes.
    Returns:
    matplotlib.axes.Axes: The axis with the plotted bounding boxes.
    """
    
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

def draw_legend(img, entries):
    """
    Draws legend labels on the top-left corner of the image.
    
    Parameters:
    img (np.ndarray): The image to draw on
    entries (list of tuples): List like [("Model A FN", "b"), ("Model B FN", "r"), ...]
    
    Returns:
    np.ndarray: Image with legend drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    margin = 10
    x = margin
    y = margin + line_height

    for text, color in entries:
        if isinstance(color, str):
            bgr = color_to_bgr(color)
        elif isinstance(color, tuple) and max(color) <= 1.0:
            r, g, b = color
            bgr = (int(b * 255), int(g * 255), int(r * 255))
        else:
            bgr = color  # assume OpenCV BGR tuple

        # Draw a small color box
        cv2.rectangle(img, (x, y - 12), (x + 12, y), bgr, -1)
        # Draw the label text
        cv2.putText(img, text, (x + 18, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height + 5

    return img