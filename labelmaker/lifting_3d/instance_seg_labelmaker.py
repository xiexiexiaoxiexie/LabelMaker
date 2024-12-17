import argparse
import logging
from pathlib import Path
from typing import Union
import cv2
import gin
import open3d as o3d
from labelmaker.label_data import get_wordnet
import numpy as np
from lifting_points import project_pointcloud
from PIL import Image
from tqdm import tqdm

import open3d as o3d
import numpy as np
mask3d_ins_prediction='/intermediate/scannet200_mask3d_2/predictions.txt'
mask3d_instances_mask='/intermediate/scannet200_mask3d_2/'
lifted_labels='/labels.txt'
mesh_name='/mesh.ply'

# get the points in array_1 that are also in array_2(coordinates)
def filter_points(array_1, array_2):
    return np.array([point for point in array_1 if np.any(np.all(point == array_2, axis=1))])
def filter_points_indices(array_1, array_2):
    # List of indices of matching points
    indices = [i for i, point in enumerate(array_1) if np.any(np.all(point == array_2, axis=1))]
    return indices
def filter_points_mask(array_1, array_2):
    # Create a boolean mask where each element is True if the point in array_1 is in array_2
    return np.array([np.any(np.all(point == array_2, axis=1)) for point in array_1])

def match_similar_category(labels_3d):
    # swivel_chair->chair
    mask_swivel_chair = labels_3d == 18
    labels_3d[mask_swivel_chair] = 2
    # desk->table
    mask_desk = labels_3d == 17
    labels_3d[mask_desk] = 8
# Load the meshes
# mesh2 = o3d.io.read_triangle_mesh("./test/table.n.02.ply")
# mesh1 = o3d.io.read_triangle_mesh("./test/mesh_yz.ply")

# # Convert the vertices of both meshes to numpy arrays
# vertices2 = np.asarray(mesh2.vertices)
# vertices1 = np.asarray(mesh1.vertices)

# # Find the intersection of the vertices
# # Using np.isin is efficient for this task
# mask = np.isin(vertices1, vertices2).all(axis=1)
# print(0)
# mask_2=filter_points_mask(vertices1, vertices2)
# print(0.5)
# # Get the indices of vertices in mesh1 that are also in mesh2
# overlapping_indices = np.where(mask)[0]
# overlapping_indices_2 = np.where(mask_2)[0]

# # Filter mesh2 vertices
# new_vertices = vertices1[overlapping_indices]
# new_vertices_2=vertices1[mask_2]
# # Update faces to keep only those referring to the remaining vertices
# # This step ensures the mesh integrity by reindexing the faces
# faces1 = np.asarray(mesh1.triangles)
# new_faces = []
# vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(overlapping_indices)}
# faces1 = np.asarray(mesh1.triangles)
# new_faces_2 = []
# vertex_map_2 = {old_idx: new_idx for new_idx, old_idx in enumerate(overlapping_indices_2)}
# print(1)
# for face in faces1:
#     if all(v in vertex_map for v in face):
#         new_faces.append([vertex_map[v] for v in face])
# print(2)
# for face in faces1:
#     if all(v in vertex_map_2 for v in face):
#         new_faces_2.append([vertex_map_2[v] for v in face])
# print(3)
# new_faces = np.array(new_faces)
# new_faces_2 = np.array(new_faces_2)

# # Create the new mesh
# new_mesh = o3d.geometry.TriangleMesh()
# new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
# new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
# print(4)
# new_mesh2 = o3d.geometry.TriangleMesh()
# new_mesh2.vertices = o3d.utility.Vector3dVector(new_vertices_2)
# new_mesh2.triangles = o3d.utility.Vector3iVector(new_faces_2)
# # Save the resulting mesh
# o3d.io.write_triangle_mesh("./test/mesh2_cut.ply", new_mesh)
# o3d.io.write_triangle_mesh("./test/mesh2_cut_2.ply", new_mesh2)

# print("mesh2_cut.ply saved!")

def filter_mask3d_instance(scene_dir, threshold, excluded_labels=None):
    
    data = np.genfromtxt(scene_dir+mask3d_ins_prediction, dtype='str', delimiter=' ')
    data_category=data[:, 1].astype(int)
    data_score = data[:, 2].astype(float)
    # filtered_data = data[(data_score > threshold) & ~np.isin(data_category,excluded_labels)]
    filtered_data = data[(data_score > threshold)]

    return filtered_data

def match_ins(scene_dir,labels_3d, filter_masks):
    for ins in filter_masks:
        dir,category, score = ins
        mask=np.loadtxt(scene_dir+mask3d_instances_mask+dir).astype(bool)
        indices=np.where(mask)[0]
        this_ins=labels_3d[mask]
        print(category)
        print(indices)

def save_mesh_from_index(scene_dir,index, new_mesh_name='/test/one_instance.ply'):
    mesh_dir=scene_dir+mesh_name
    print(f"loading mesh from {mesh_dir}")
    mesh = o3d.io.read_triangle_mesh(mesh_dir)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)
    new_vertices = mesh_vertices[index]
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    print(f"!!!!!!!!!!!index_shape is {index.shape}saving mesh with {new_vertices.shape[0]} vertices, in {scene_dir+new_mesh_name}")
    o3d.io.write_triangle_mesh(scene_dir+new_mesh_name, new_mesh)


def match_single_category_ins(scene_dir,category_pc_index,labels_3d, filter_masks,category_name,b_test=False):
    # from a single category pointcloud index and all the instance masks, instance seg the pointcloud index
    used_index=np.zeros(category_pc_index.shape[0])
    instance_id=1
    for ins in filter_masks:
        # category in scannet200 space
        dir,category, score = ins
        mask=np.loadtxt(scene_dir+mask3d_instances_mask+dir).astype(bool)
        mask3d_indices=np.where(mask)[0]
        # print(f"mask3d_indices: {mask3d_indices}")
        # this_ins_index=labels_3d[mask]
        # print(f"this mask has {mask3d_indices.shape[0]} points")
        # print(np.isin(this_ins_index,category_pc_index))

        category_pc_index[used_index.astype(bool)]=-1

        overlapping_points_num=np.sum(np.isin(category_pc_index,mask3d_indices))
        overlapping_category_index=category_pc_index[np.isin(category_pc_index,mask3d_indices)]

        mask_overlapping_percentage=overlapping_points_num/mask3d_indices.shape[0] # the more the better
        if overlapping_points_num>10:
            print(f"current_mask: {dir}, overlapping_points_num: {overlapping_points_num}, mask3d_indices: {mask3d_indices.shape[0]}, overlapping_percentage: {mask_overlapping_percentage}")
        if mask_overlapping_percentage<0.4:
            # mask is not overlapping with the category pointcloud too much, wrong mask
            continue
        used_index[np.isin(category_pc_index,mask3d_indices)]=1

        total_points_percentage_1=overlapping_points_num/category_pc_index.shape[0] # the more the better

        if mask_overlapping_percentage>0.05:
            print(f"in mask category: {category}, there are {overlapping_points_num} points intersect with the category pointcloud, percentage1: {mask_overlapping_percentage}, percentage2: {mask_overlapping_percentage*total_points_percentage_1}, percentage3: {mask_overlapping_percentage*total_points_percentage_1*total_points_percentage_1}")
        # test cushion, in scanet200 space
        # if int(category)==13 and b_test and overlapping_points_num>0:
        #     save_mesh_from_index(scene_dir,mask3d_indices)
        if mask_overlapping_percentage>0.5:
            save_mesh_from_index(scene_dir,overlapping_category_index, new_mesh_name='/test/one_instances/'+category_name+'_'+str(instance_id)+'.ply')
            instance_id+=1

        # np.isin(this_ins_index,category_pc_index).all(axis=0)
def get_wordnet_name_by_id(data, target_id):
    for record in data:
        if record['id'] == target_id:
            return record['name']
    return None 

def match_category_ins(scene_dir,labels_3d, filter_masks,excluded_labels):
    # labels_3d in wordnet space
    print(f"labels_3d unique: {np.unique(labels_3d)}")
    b_test=False
    for category in np.unique(labels_3d):
        
        category_index=np.where(labels_3d==category)
        save_mesh_from_index(scene_dir,category_index[0], new_mesh_name='/test/'+get_wordnet_name_by_id(get_wordnet(),category)+'.ply')
        if category in excluded_labels:
            continue
        # if category_index[0].shape[0]<100:
        #     print(f"category {category}: {get_wordnet_name_by_id(get_wordnet(),category)} has less than 100 points, skip")
        #     continue
        category_name=get_wordnet_name_by_id(get_wordnet(),category)
        print("-----------------")
        print(f"handling category: {category}: {category_name}, single_category_pointcloud_shape{category_index[0].shape}")
        # wordnet space 12:cushion
        if category==12:
            b_test=True
        match_single_category_ins(scene_dir,category_index[0],labels_3d, filter_masks,category_name,b_test)
        b_test=False
        
        

def prepare_labelmaker_labels(scene_dir):
    labels_3d=np.loadtxt(scene_dir+lifted_labels)
    labels_3d=labels_3d.astype(int)
    match_similar_category(labels_3d)
    return labels_3d

    
   
def main(scene_dir: Union[str, Path],  output_file: str, output_mesh: str):
    excluded_labels=[1,6,5,9,15,16] #wordnet200 space
    # excluded_labels=[0] #wordnet200 space

    filter_masks=filter_mask3d_instance(scene_dir, threshold=0)
    print(filter_masks)
    labels_3d=prepare_labelmaker_labels(scene_dir)
    match_category_ins(scene_dir,labels_3d, filter_masks,excluded_labels)
    # lifted_labels_array=np.loadtxt(scene_dir+lifted_labels)


def arg_parser():
  parser = argparse.ArgumentParser(
      description=
      'Project 3D points to 2D image plane and aggregate labels and save label txt'
  )
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help=
      'Path to workspace directory. There should be a "color" folder inside.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='labels.txt',
      help='Name of files to save the labels',
  )
  parser.add_argument(
      '--output_mesh',
      type=str,
      default='point_lifted_mesh.ply',
      help='Name of files to save the labels',
  )
  
  return parser.parse_args()
def temp_main(scene_dir: Union[str, Path],  output_file: str, output_mesh: str):
    for file_id in ['004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040']:
    
        mask_file='/home/xiefujing/LabelMaker/apple_scanner/2024_11_15_14_10_27_office/output/intermediate/scannet200_mask3d_2/pred_mask/'+file_id+'.txt'
        mask_index=np.where(np.loadtxt(mask_file).astype(bool))[0]
        save_mesh_from_index(scene_dir,mask_index, new_mesh_name='/test/'+file_id+'_one_instance.ply')
def temp_temp_main(file_name):
    data=np.loadtxt(file_name).astype(int)
    numof2=np.sum(data==2)
    print(f"num of 2: {numof2}")

def count_mesh_color(file_name):
    mesh = o3d.io.read_triangle_mesh(file_name)

    # Ensure the mesh has vertex colors
    if not mesh.has_vertex_colors():
        print("Mesh does not contain vertex colors.")
    else:
        # Convert vertex colors to a numpy array
        vertex_colors = np.asarray(mesh.vertex_colors)

        # Define the target color (188, 189, 34)
        target_color = np.array([188/255.0, 189/255.0, 34/255.0])  # Open3D expects normalized values (0-1)

        # Find the indices of the vertices with the target color
        color_match_indices = np.all(np.abs(vertex_colors - target_color) < 1e-6, axis=1)

        # Count how many points have the target color
        num_matching_points = np.sum(color_match_indices)

        print(f"Number of points with color (188, 189, 34): {num_matching_points}")
if __name__ == '__main__':
  args = arg_parser()
  main(
      scene_dir=args.workspace,
      output_file=args.output,
      output_mesh=args.output_mesh,
  )
#   temp_main(
#       scene_dir=args.workspace,
#       output_file=args.output,
#       output_mesh=args.output_mesh,
#       )
#   temp_temp_main('/home/xiefujing/LabelMaker/apple_scanner/2024_11_15_14_10_27_office/output/labels.txt')
#   count_mesh_color('/home/xiefujing/LabelMaker/apple_scanner/2024_11_15_14_10_27_office/output/point_lifted_mesh.ply')
