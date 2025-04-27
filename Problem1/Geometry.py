# %%
import sys
import time, os
os.environ["TZ"] = "US/Eastern"
time.tzset()
import numpy as np
import torch
import yaml
import cv2
import matplotlib.pyplot as plt

!pip install -q open3d
import open3d as o3d

# !pip install --upgrade plotly 1>/dev/null
%matplotlib inline

# %%
from Problem1 import *
from utils import *
hello()

# %% [markdown]
# ### Data Visualization
# 
# First we will load some files from the data. Check out the data folder to get familiar with some of the files. The yaml file contains information about which semantic category each integer the label corresponds to.

# %%
# Load Semantic KITTI

DATA_PATH = "Data"
velodyne_dir = os.path.join(DATA_PATH, 'velodyne')
label_dir = os.path.join(DATA_PATH, 'labels')

frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]
velodyne_list = ([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
label_list = ([os.path.join(label_dir, str(frame).zfill(6)+'.label') for frame in frames_list])

# Label map
config_file = os.path.join(DATA_PATH, "semantic_kitti.yaml")
kitti_config = yaml.safe_load(open(config_file, 'r'))
LABELS_REMAP = kitti_config["learning_map"]

# %% [markdown]
# First, we will visualize a single frame of the data. This will take a few seconds to load. Once it does, move your mouse to take a look at the scene. See the gap in the middle? That is where the ego-vehicle was located. Next, let's see if we can use ICP to aggregate frames and fill the map.

# %%
xyz_source, label_source = get_cloud(velodyne_list, label_list, 0, LABELS_REMAP)
plot_cloud(xyz_source, label_source)

# %% [markdown]
# # Registration
# Open the **Problem1.py** python file, and we will use `register_clouds()` and `combine_clouds()`. `register_clouds()` is where we will perform ICP on the raw point clouds given noisy initial guesses from odometry, and `combine_clouds()` allows us to combine sequential frames using a registration matrix. 

# %%
# Register point clouds
start = 0

# Load first point cloud in sequence
xyz_source, label_source = get_cloud(velodyne_list, label_list, start, LABELS_REMAP)
xyz_prev = xyz_source
label_prev = label_source
odometry = np.loadtxt(os.path.join(DATA_PATH, 'odometry.txt'))
icp_transforms = []

# Loop through next 10 point clouds
for i in range(start+1, start+10):
  xyz_target, label_target = get_cloud(velodyne_list, label_list, i, LABELS_REMAP)
  # Acquire initial matrix
  init_mat = get_init_mat(odometry, i)
  
  # Estimate regisration matrix
  reg_mat = register_clouds(xyz_prev, xyz_target, trans_init=init_mat)
  icp_transforms.append(reg_mat)
  
  # Combine the point clouds
  xyz_source, label_source = combine_clouds(xyz_source, xyz_target, label_source, label_target, reg_mat)
  xyz_prev = xyz_target

# Plot registered clouds
plot_cloud(xyz_source, label_source)

# %% [markdown]
# Looks great! However, now there are traces left behind by moving vehicles. This could be throwing off our ICP, so implement the function `mask_static()` to remove dynamic object points from the point cloud using their labels. Also implement `mask_dynamic()` to only return the points corresponding to dynamic classes. Check the yaml file to identify the integer corresponding to the car and bus classes, and remove from the input point clouds.

# %%
# Register point clouds
start = 0
xyz_source, label_source = get_cloud(velodyne_list, label_list, start, LABELS_REMAP)
xyz_prev = xyz_source
label_prev = label_source
odometry = np.loadtxt(os.path.join(DATA_PATH, 'odometry.txt'))
icp_transforms = []
for i in range(start+1, start+10):
  xyz_target, label_target = get_cloud(velodyne_list, label_list, i, LABELS_REMAP)
  # remove dynamic objects
  static_prev, __ = mask_static(xyz_prev, label_prev)
  static_target, __ = mask_static(xyz_target, label_target)
  init_mat = get_init_mat(odometry, i)
  # Register without dynamic objects
  reg_mat = register_clouds(static_prev, static_target, trans_init=init_mat)
  icp_transforms.append(reg_mat)
  xyz_source, label_source = combine_clouds(xyz_source, xyz_target, label_source, label_target, reg_mat)
  xyz_prev = xyz_target
  label_prev = label_target
plot_cloud(xyz_source, label_source)

# %% [markdown]
# # Instance Segmentation
# Looks better, however there are still traces. Let's take a closer look at the moving vehicles by creating a mask.

# %%
# Get Instances
def get_instances(xyz, label):
  road_mask = (xyz[:, 1] <= 13) & (xyz[:, 1] >= -4)
  xyz = xyz[road_mask, :]
  label = label[road_mask]
  xyz_dynamic, label_dynamic = mask_dynamic(xyz, label)
  return xyz_dynamic, label_dynamic

# Get 10th point cloud
xyz_10, label_10 = get_cloud(velodyne_list, label_list, 10, LABELS_REMAP)
# Only plot cars on road
xyz_dynamic, label_dynamic = get_instances(xyz_10, label_10)
plot_cloud(xyz_dynamic, label_dynamic)

# %% [markdown]
# There seem to be several distinct clusters of points belonging to instances of vehicles on the road. Implement functions `cluster_dists()`, `new_centroids()`, and `num_instances()` for the kMeans algorithm in the python file to identify instances in an unsupervised manner. Pay attention to how many distinct instances you see, this will be important for initialization of the algorithm.

# %%
__, clustered_labels = cluster(xyz_dynamic)
# Plot clusters
plot_cloud(xyz_dynamic, clustered_labels+1)

# %% [markdown]
# Next we will visualize the same scene, however without traces.

# %%
# Create scene without traces
start = 0
xyz_source, label_source = get_cloud(velodyne_list, label_list, start, LABELS_REMAP)
xyz_prev = xyz_source
label_prev = label_source
odometry = np.loadtxt(os.path.join(DATA_PATH, 'odometry.txt'))
for i in range(start+1, start+10):
  xyz_target, label_target = get_cloud(velodyne_list, label_list, i, LABELS_REMAP)
  static_prev, __ = mask_static(xyz_prev, label_prev)
  static_target, __ = mask_static(xyz_target, label_target)
  reg_mat = icp_transforms[i-1]
  xyz_source, label_source = combine_clouds(xyz_source, xyz_target, label_source, label_target, reg_mat)
  xyz_prev = xyz_target
  label_prev = label_target

xyz_static, label_static = mask_static(xyz_source, label_source)
xyz_10, label_10 = get_cloud(velodyne_list, label_list, 10, LABELS_REMAP)
xyz_dynamic, __ = get_instances(xyz_10, label_10)
__, label_dynamic = cluster(xyz_dynamic)
xyz_static, label_static = downsample_cloud(xyz_static, label_static, 100000 - label_dynamic.shape[0])
xyz_all, label_all = combine_clouds(xyz_static, xyz_dynamic, label_static, label_dynamic, np.eye(4))

plot_cloud(xyz_all, label_all)

# %% [markdown]
# # LiDAR to Camera Transformations
# There are distinct advantages and disadvantages of each sensor. In adverse driving conditions, it is even more important to leverage a full sensor suite. Here, we will be transforming LiDAR points to pixels on data from the off-road driving dataset RELLIS-3D.
# 
# First, we will visualize the image without LiDAR points below.

# %%
image = cv2.imread(os.path.join("Data", "RELLIS", "Camera.jpg"))
res = print_projection_plt(image=image)

plt.subplots(1,1, figsize = (20,20) )
plt.title("Camera without Velodyne")
plt.imshow(res)

# %% [markdown]
# Next, we invoke `to_pixels()` which will receive 3D LiDAR points (Nx3), the intrinsic matrix, and a transformation from LiDAR frame to camera frame. Return the pixel values as a Nx2 matrix. Also return the depth for each point by transforming to camera frame, then obtaining depth from the third column. If this works correctly, you should see points which generally match the layout of the scene. Note that LiDAR is sparseand sensitive to noise from vegetation, so the points may not match up exactly.

# %%
# Camera intrinsic matrix
P = np.array([[2.81364327e+03, 0.00000000e+00, 9.69285772e+02],
              [0.00000000e+00, 2.80832608e+03, 6.24049972e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Transform from LiDAR to camera
RT = np.array([[0.03462247,  0.99936055, -0.00893175, -0.03566209],
               [ 0.00479177, -0.00910301, -0.99994709, -0.17154603],
               [-0.99938898,  0.03457784, -0.00510388, -0.13379309],
               [ 0.,          0.,          0.,          1.        ]])

fpath = os.path.join("Data", "RELLIS", "pc.bin")
pc = np.fromfile(fpath, dtype=np.float64).reshape(-1, 3)
imgpoints, d = to_pixels(pc, P, RT)

# Convert depth to color and remove invalid points
imgpoints = imgpoints.T
mask = (imgpoints[0, :] > 0) & (imgpoints[1, :] > 0)
imgpoints = imgpoints[:, mask]
d = d[mask]
c_ = depth_color(d)

# Plot image
res = print_projection_plt(image, points=imgpoints, color=c_)
plt.subplots(1,1, figsize = (20,20) )
plt.title("Velodyne points to camera image Result")
plt.imshow(res)


