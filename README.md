# LiDAR Data Processing for Ego-Vehicle Perception 

## Project Overview

In this project, we explore how 3D LiDAR point cloud data captured from an ego vehicle is processed and interpreted for downstream perception tasks.
The following key tasks were implemented:

- **Point Cloud Registration**:
  Align consecutive LiDAR scans over time using Iterative Closest Point (ICP) algorithms implemented through Open3D.

- **Dynamic vs Static Segmentation**:
  Separate moving objects (like cars and buses) from static environment features (like roads, buildings) using label-based masking.

- **K-Nearest Neighbor (KNN) Based Clustering**:
  Perform unsupervised clustering on dynamic points to identify individual instances of vehicles in the scene.

- **LiDAR-to-Camera Projection**:
  Project 3D LiDAR points onto the 2D image plane by applying extrinsic and intrinsic transformations to simulate multi-modal sensor fusion.

- **Custom K-Means Implementation**:
  Developed a custom K-Means clustering algorithm to segment detected vehicles, and benchmarked it against the scikit-learn KMeans implementation.

- **Point Cloud Fusion**:
  Fused multiple registered point clouds to build a richer spatial understanding of the surrounding environment.

---

## Technologies Used
- Python
- Open3D (for point cloud operations)
- scikit-learn (for clustering)
- NumPy (for numerical operations)
- Matplotlib (for visualization if needed)
- LiDAR sensor data (processed offline)

---

## Project Structure
```plaintext
.
├── Geometry.ipynb      # Jupyter Notebook: Exploration, visualization, and full pipeline execution
├── Problem1.py         # Python Script: Core functions for registration, masking, clustering, and projection
├── README.md           # Project description and instructions
└── Data/               # (Optional) Directory for point cloud and calibration data
```

---

## How to Run
1. Install required Python packages:
```bash
pip install open3d scikit-learn numpy matplotlib
```

2. Open the `Geometry.ipynb` notebook and run through the cells sequentially.

3. Alternatively, import and use individual functions from `Problem1.py` inside your own Python scripts.

---

## Key Results
- Successfully aligned point clouds from sequential frames with high fitness using ICP.
- Isolated dynamic vehicles and identified distinct instances through custom clustering.
- Projected 3D points accurately onto camera images, enabling multi-sensor fusion.

---

## Summary of Core Functions
| Module                  | Purpose |
|:-------------------------|:--------|
| `register_clouds`        | Aligns two point clouds using ICP |
| `combine_clouds`         | Merges transformed source and target clouds |
| `mask_dynamic`, `mask_static` | Segregates moving and static objects |
| `cluster`, `cluster_sci` | Performs clustering to identify instances |
| `to_pixels`              | Projects 3D LiDAR points into 2D camera image space |

---

## Acknowledgments

The codebase builds on top of Open3D and Scikit-learn libraries and is inspired by core LiDAR processing principles used in autonomous vehicle perception pipelines.

---

## Short Description

> Processing and interpreting ego-vehicle LiDAR data for registration, dynamic segmentation, clustering, and LiDAR-camera fusion.