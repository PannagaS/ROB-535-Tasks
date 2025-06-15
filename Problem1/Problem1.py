"""
Code adapted from Joey Wilson, 2023; Modified by Pannaga Sudarshan, 2024.
"""

import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans


# Given two Nx3 and Mx3 point clouds and initial SE(3) matrix
# Register using open3d and return SE(3) registration matrix
# Refer to open3d geometry.PointCloud
# As well as open3d registration.registration_icp
# See the following tutorial for help
# http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
# Grade will be based on the fitness score compared to our solution
def register_clouds(xyz_source, xyz_target, trans_init=None):
  if trans_init is None:
    trans_init = np.eye(4)
  threshold = 1e-1
  max_iters = 100

  # create o3d pointclouds for the source and target
  source_pcd = o3d.geometry.PointCloud()
  target_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(xyz_source)
  target_pcd.points = o3d.utility.Vector3dVector(xyz_target)
  
  source = source_pcd
  target = target_pcd

  # Pre-registration similarity
  evaluation_pre = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
  print("Before registration:", evaluation_pre)

  # register the point clouds using registration_icp
  reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters)
    )

  evaluation_post = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, reg_p2p.transformation)
  print("After registration:", evaluation_post)

  # obtain the transformation matrix from reg_p2p
  reg_mat = reg_p2p.transformation
  return reg_mat


# Given two point clouds Nx3, corresponding N labels, and registration matrix
# Transform the points of the source (time T-1) to the pose at frame T
# Return combined points (N+M)x3 and labels (N+M)
def combine_clouds(xyz_source, xyz_target, labels_source, labels_target, reg_mat):
  # TODO: Apply transformation matrix on xyz_source
  source_pcd = o3d.geometry.PointCloud()
  target_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(xyz_source)
  target_pcd.points = o3d.utility.Vector3dVector(xyz_target)

  # xyz_transformed = source_pcd.transform(reg_mat)
  
  # # TODO: Concatenate xyz and labels
  # # Note: Concatenate transformed followed by target
  # xyz_all = xyz_transformed + target_pcd #xyz_all is o3d object
  # xyz_all = np.asarray(xyz_all.points) #convert back to np


  source_pcd.transform(reg_mat)
    
  # Convert point clouds to numpy arrays and concatenate
  xyz_transformed = np.asarray(source_pcd.points)
  xyz_target = np.asarray(target_pcd.points)
  xyz_all = np.concatenate((xyz_transformed, xyz_target))
  



  label_all = np.concatenate((labels_source, labels_target))
  
  return xyz_all, label_all


# Mask to return points in only the vehicle and bus class
# Check the yaml file to identify which labels correspond to vehicle and bus
# Note that labels are mapped to training set, meaning in range 0 to 20
def mask_dynamic(xyz, label):
  # TODO: Create a mask such that label == vehicle or label == bus
  # 13 is bus, 10 is car
  
  
  # for point in range(len(label)):
  #   if label[point] == 13 or label[point] == 10:
  #     print("yes")
  #     dynamic_mask.append(point)
  # dynamic_mask =  np.isin(label, [10,11,13,15,18])
  dynamic_mask = (label == 1) | (label == 5)
  return xyz[dynamic_mask, :], label[dynamic_mask]
  


# Similarly, mask out the vehicle and bus class to return only the static points
def mask_static(xyz, label):
  # TODO: Create a mask opposite of above
  # static_mask = []
  # static_objects = [0, 1, 16]
  # static_mask = np.isin(label, static_objects)
  notStatic = (label == 1) | (label == 5) 
  static_mask = ~notStatic
  # for point in range(len(label)):
  #   if label[point] != 13 or label[point]!= 10:
  #     static_mask.append(point)
  return xyz[static_mask, :], label[static_mask]


# Given an Nx3 matrix of points and a Cx3 matrix of clusters
# Return for each point the index of the nearest cluster
# For efficiency, useful functions include np.tile, np.linalg.norm, and np.argmin  
#from IPython import embed
def cluster_dists(xyz, clusters):
  #print(clusters)
  #embed()
  N, _ = xyz.shape
  xyz = xyz.reshape(N, 1, 3)
  C = clusters.shape[0]
  closest_clusts = np.zeros(N)

  #embed()
  # TODO: Create assignments between each point and the closest cluster
  # 
  for point in range(N):
    dist = np.zeros(C)
    for cluster in range(C):
      
      dist[cluster] = (np.linalg.norm(xyz[point] - clusters[cluster]))
      #embed()
    closest_clusts[point] = int(np.argmin(dist))
  
  #closest_clusts = closest_clusts.astype(int).tolist()
  return closest_clusts


# Given Nx3 points and N assignments (for each point, the index of the cluster)
# Calculate the new centroids of each cluster
# Return centroids shape Cx3
def new_centroids(xyz, assignments, C):
  new_instances = np.zeros((C, 3))
  # TODO: Calculate new clusters by the assignments
  #embed()
  clusters_points = [np.empty((0, 3)) for _ in range(C)]
  #embed()
  # Append each point to the correct cluster array based on assignments
  for i in range(len(xyz)):
      cluster_idx = assignments[i]
      #embed()
      clusters_points[int(cluster_idx)] = np.vstack([clusters_points[int(cluster_idx)], xyz[i]])
  
  
  #embed()
  for i in range(len(clusters_points)):
      new_instances[i] = np.mean(clusters_points[i],axis =0)
  #embed()

  return new_instances


# Returns an integer corresponding to the number of instances of dynamic vehicles
# Shown in the point cloud
def num_instances():
  # TODO: Return the number of instances
  number_of_point_cloud_instances_i_see = 5
  return number_of_point_cloud_instances_i_see


# K means algorithm. Given points, calculates and returns clusters.
def cluster(xyz):
  #print(xyz)
  C = num_instances()
  rng = np.random.default_rng(seed=1)
  instances = xyz[rng.choice(xyz.shape[0], size=C, replace=False), :]
  prev_assignments = rng.choice(C, size=xyz.shape[0])
  while True:
    assignments = cluster_dists(xyz, instances)
    instances = new_centroids(xyz, assignments, C)
    #prev_assignments = prev_assignments.astype(int).tolist()
    if (assignments == prev_assignments).all():
      return instances, assignments
    prev_assignments = assignments


# Sci-kit learn implementation which is more advanced. 
# Try changing the random seed in cluster (will not be tested) and observe 
# how the clusters change. 
# Then try running scikit learn clustering with different random states. 
def cluster_sci(xyz):
  kmeans = KMeans(n_clusters=num_instances(), random_state=5, n_init="auto").fit(xyz)
  clustered_labels = kmeans.predict(xyz)
  return kmeans.cluster_centers_, clustered_labels

# Given Nx3 point cloud, 3x3 camera intrinsic matrix, and 4x4 LiDAR to Camera 
# Compute image points Nx2 and depth in camera-frame per point
def to_pixels(xyz, P, RT):
  # Convert to camera frame
  transformed_points = np.zeros_like(xyz)
  #embed()

  for point in range(xyz.shape[0]):
   
    transformed_points[point] = ((RT @ np.hstack((xyz[point], [1])).reshape(4,1))[:3]).flatten()
  
  #embed()
  N = transformed_points.shape[0]
  # Depth in camera frame
  d = np.zeros(N)
  for point in range(N):
    d[point] = transformed_points[point][2]
  #embed()
  # Use intrinsic matrix
  image_points = np.zeros((N,3))
 
  for point in range(N):
    image_points[point] = P @ transformed_points[point]
  
  # Normalize 
  image_x = image_points[:,0]/image_points[:,2]
  image_y = image_points[:,1]/image_points[:,2]

  imgpoints = np.stack([image_x, image_y], axis=1)
  return imgpoints, d
