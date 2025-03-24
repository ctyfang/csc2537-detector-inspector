import os
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from utils import plot_boxes, plot_pred_oriented_bboxes, draw_legend

"""
Run
`python diff_plot.py --mode fp` to highlight false positive objects
`python diff_plot.py --mode fn` to highlight false negative objects
"""

def compute_iou(box1, box2):
    """Compute IoU for 2D boxes: [x1, y1, x2, y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def filter_boxes_by_iou(target_boxes, reference_boxes, iou_threshold, mode='fp'):
    """
    For 'fp': return boxes in target_boxes that have no match in reference_boxes.
    For 'fn': return boxes in reference_boxes not matched by any box in target_boxes.
    """
    target_boxes = np.atleast_2d(target_boxes)
    reference_boxes = np.atleast_2d(reference_boxes)
    unmatched = []

    for tbox in (target_boxes if mode == 'fp' else reference_boxes):
        match_found = False
        for rbox in (reference_boxes if mode == 'fp' else target_boxes):
            iou = compute_iou(tbox[1:5], rbox[1:5])  # Use only x1,y1,x2,y2
            if iou > iou_threshold:
                match_found = True
                break
        if not match_found:
            unmatched.append(tbox)

    return np.array(unmatched)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fp', 'fn'], required=True, help="fp (false positives) or fn (false negatives)")
    parser.add_argument('--index', type=int, default=1, help="Index of the frame to visualize")
    parser.add_argument('--threshold', type=float, default=0.5, help="IoU threshold")
    args = parser.parse_args()

    label_dir = 'D:/Master/Courses/Information Visualization/csc2537-detector-inspector/data/labels'
    image_dir = os.path.join(label_dir, "..", "images")
    model_dir = os.path.join(label_dir, "..", "model_outputs")
    model_names = ["fcos3d", "fcos3d_finetune"]
    colors = ['b', 'r']

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    filename = label_files[args.index]
    gt = np.atleast_2d(np.loadtxt(os.path.join(label_dir, filename), delimiter=','))
    img = cv2.imread(os.path.join(image_dir, filename.replace('.txt', '.jpg')))
    preds = [np.atleast_2d(np.loadtxt(os.path.join(model_dir, m, filename), delimiter=',')) for m in model_names]

    if args.mode == 'fp':
        for i, pred in enumerate(preds):
            fp = np.atleast_2d(filter_boxes_by_iou(pred, gt, args.threshold, mode='fp'))
            if fp.size > 0:
                img = plot_boxes(img, fp, colors[i], f"{model_names[i]} FP")

        legend_entries = [("Model A FP", 'b'), ("Model B FP", 'r')]
        img = draw_legend(img, legend_entries)
        cv2.imwrite("false_positive_image.png", img)

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_title("False Positives in BEV")
        for i, pred in enumerate(preds):
            fp = np.atleast_2d(filter_boxes_by_iou(pred, gt, args.threshold, mode='fp'))
            if fp.size > 0:
                ax = plot_pred_oriented_bboxes(fp[:, 5:], ax, colors[i], alpha=0.6)

        for radius in [50, 100, 150]:
            circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--')
            ax.add_patch(circle)

        ax.set_xlim([-50, 50])
        ax.set_ylim([0, 125])
        ax.set_title("False Positives in BEV")

        legend_entries = [mpatches.Patch(color='blue', label='Model A FP'), mpatches.Patch(color='red', label='Model B FP')]
        ax.legend(handles=legend_entries, loc='upper right', fontsize='small', framealpha=0.8)

        plt.savefig("false_positive_bev.png")
        plt.close()

    elif args.mode == 'fn':
        fn_a = np.atleast_2d(filter_boxes_by_iou(preds[0], gt, args.threshold, mode='fn'))
        fn_b = np.atleast_2d(filter_boxes_by_iou(preds[1], gt, args.threshold, mode='fn'))

        def same_box(b1, b2, threshold=0.999):
            if b1.shape[0] < 5 or b2.shape[0] < 5:
                return False
            return compute_iou(b1[1:5], b2[1:5]) > threshold

        missed_by_both = []
        missed_by_a_only = []
        missed_by_b_only = []

        if fn_a.size > 0 and fn_b.size > 0:
            for box in fn_a:
                if any(same_box(box, b2) for b2 in fn_b):
                    missed_by_both.append(box)
                else:
                    missed_by_a_only.append(box)

            for box in fn_b:
                if not any(same_box(box, b1) for b1 in fn_a):
                    missed_by_b_only.append(box)

            missed_by_a_only = np.atleast_2d(np.array(missed_by_a_only))
            missed_by_b_only = np.atleast_2d(np.array(missed_by_b_only))
            missed_by_both   = np.atleast_2d(np.array(missed_by_both))
        else:
            missed_by_both = np.empty((0, 8))
            missed_by_a_only = fn_a if fn_a.size > 0 else np.empty((0, 8))
            missed_by_b_only = fn_b if fn_b.size > 0 else np.empty((0, 8))

        # Image view
        if missed_by_a_only.size > 0:
            img = plot_boxes(img, missed_by_a_only, 'b', "Model A FN")
        if missed_by_b_only.size > 0:
            img = plot_boxes(img, missed_by_b_only, 'r', "Model B FN")
        if missed_by_both.size > 0:
            img = plot_boxes(img, missed_by_both, (0, 255, 255), "Both Models FN")  # Yellow

        legend_entries = [("Model A FN", 'b'), ("Model B FN", 'r'), ("Both Models FN", (0, 255, 255))]
        img = draw_legend(img, legend_entries)
        cv2.imwrite("false_negative_image.png", img)

        # BEV view
        fig, ax = plt.subplots(figsize=(8, 10))
        # ax = plot_oriented_boxes_bev(ax, gt, 'g', 'Ground Truth')
        if missed_by_a_only.size > 0:
            ax = plot_pred_oriented_bboxes(missed_by_a_only[:, 5:], ax, 'b', alpha=0.6)
        if missed_by_b_only.size > 0:
            ax = plot_pred_oriented_bboxes(missed_by_b_only[:, 5:], ax, 'r', alpha=0.6)
        if missed_by_both.size > 0:
            ax = plot_pred_oriented_bboxes(missed_by_both[:, 5:], ax, (1.0, 1.0, 0.0), alpha=0.6)  # Yellow

        for radius in [50, 100, 150]:
            circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--')
            ax.add_patch(circle)

        ax.set_xlim([-50, 50])
        ax.set_ylim([0, 125])
        ax.set_title("False Negatives in BEV")

        legend_entries = [mpatches.Patch(color='blue', label='Model A FN'), mpatches.Patch(color='red', label='Model B FN'), mpatches.Patch(color='yellow', label='Both Models FN')]
        ax.legend(handles=legend_entries, loc='upper right', fontsize='small', framealpha=0.8)
        plt.savefig("false_negative_bev.png")
        plt.close()


if __name__ == '__main__':
    main()
