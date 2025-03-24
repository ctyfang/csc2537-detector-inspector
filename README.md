# csc2537-detector-inspector
Project repository for CSC2537 class project: Detector Inspector, a tool for comparing outputs from 3D detection models.

Features:
- Multimodal, multiview plots for inspecting model differences in Bird's-Eye-View and Image-View
- Diff plots to focus exploration on model differences

GT Data Format
The data folder has the following structure:

```
data/
├── calibration/
│   ├── calib1.txt
│   ├── calib2.txt
│   └── ...
├── images/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── labels/
│   ├── label1.txt
│   ├── label2.txt
│   └── ...
└── model_outputs/
    ├── model1/
    │   ├── output1.txt
    │   ├── output2.txt
    │   └── ...
    └── model2/
        ├── output1.txt
        ├── output2.txt
        └── ...
```

- `calibration/`: Contains calibration files for the sensors.
- `images/`: Contains image files used for detection.
- `labels/`: Contains ground truth label files. Each row is formatted as `label_class, [x1, y1, x2, y2], [x, y, z, l, w, h, yaw]`
- `model_outputs/`: Contains output files from different models, organized by model name. The outputs are formatted in the same way as the labels. Sample data was generated using MMDetection3D with the two publicly available. FCOS3D model checkpoints
