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

# python ./labelmaker/lifting_3d/get_objects_in_consensus_mesh.py --workspace $WORKSPACE_DIR
logging.basicConfig(level="INFO")
log = logging.getLogger('Fujing: seperate mesh by consensus class')

def load_mesh(filename):
    return o3d.io.read_triangle_mesh(filename)
def get_class_name(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # On left mouse button click
        # Retrieve the original depth value (before normalization)
        original_value = param[y, x]
        color_map = {}
        for item in get_wordnet():
          color_map[item['id']] = item['name']
        # Get the class name from the dictionary, if available
        print(f"Class name: {color_map[original_value]}")
        return original_value
def find_unique_colors(mesh):
    # Extract unique colors from the mesh vertex colors
    vertex_colors = np.asarray(mesh.vertex_colors)
    unique_colors = np.unique(vertex_colors, axis=0)
    return unique_colors


def extract_class_vertices(lifted_mesh, color_mesh, target_color):
    class_colors = np.asarray(lifted_mesh.vertex_colors)
    true_vertices = np.asarray(color_mesh.vertices)
    true_colors = np.asarray(color_mesh.vertex_colors)
    triangles = np.asarray(color_mesh.triangles)
    vertex_mask = np.all(np.abs(class_colors - target_color) < 0.01, axis=1)

    # Find triangles that are fully within the masked vertices
    triangle_mask = np.all(vertex_mask[triangles], axis=1)
    class_triangles = triangles[triangle_mask]

    # Get the unique vertices used in the selected triangles
    unique_vertex_indices = np.unique(class_triangles)
    class_vertices = true_vertices[unique_vertex_indices]
    class_colors_true = true_colors[unique_vertex_indices]

    # Remap triangle indices to match the new set of vertices
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
    remapped_triangles = np.vectorize(remap.get)(class_triangles)

    # Create a new mesh for the current color class
    class_mesh = o3d.geometry.TriangleMesh()
    class_mesh.vertices = o3d.utility.Vector3dVector(class_vertices)
    class_mesh.vertex_colors = o3d.utility.Vector3dVector(class_colors_true)
    class_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    return class_mesh

def exact_color(lifted_mesh, color_mesh, scene_dir,target_color=np.array([0.5372549, 0.24705882, 0.05490196])):
    lifted_vertices = np.asarray(lifted_mesh.vertices)
    lifted_colors = np.asarray(lifted_mesh.vertex_colors)
    lifted_faces = np.asarray(lifted_mesh.triangles)
    color_faces = np.asarray(color_mesh.triangles)
    color_vertices = np.asarray(color_mesh.vertices)
    color_colors = np.asarray(color_mesh.vertex_colors)

    # Define the target color (e.g., 100, 100, 100)
    tolerance = 0.005  # Define tolerance for color matching
    
    # Find vertices that match the target color within the tolerance
    color_diff = np.linalg.norm(lifted_colors - target_color, axis=1)
    matching_vertices_indices = np.where(color_diff < tolerance)[0]
    print(f"Found {len(matching_vertices_indices)} matching vertices")
    print(f"matching_vertices_indices: {matching_vertices_indices}")
    matching_colors = lifted_colors[matching_vertices_indices]
    print(f"Found {len(matching_colors)} matching matching_colors")
    print(f"matching_colors: {matching_colors}")
    if len(matching_vertices_indices) > 0:
        # Extract the matching vertices and faces
        matching_vertices = color_vertices[matching_vertices_indices]
        
        # Create a new mesh for the extracted part
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(matching_vertices)
        
        # Find faces that have vertices from the matching set
        new_faces = []
        for face in color_faces:
            if all(vertex in matching_vertices_indices for vertex in face):
                new_faces.append(face)
        
        # Convert new_faces to an array
        new_faces = np.array(new_faces)
        
        new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
        
        # Optionally assign the same color to the new mesh vertices
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(matching_colors)
        
        # Visualize the result
        # o3d.visualization.draw_geometries([new_mesh])
        
        # Optionally save the new mesh
        o3d.io.write_triangle_mesh(f"{str(scene_dir)}/test_single_color_mesh.ply", new_mesh)
def save_class_meshes(lifted_mesh, color_mesh, scene_dir):
    # Get unique colors representing each class in the classified mesh
    unique_colors = find_unique_colors(lifted_mesh) # 0-1 range
    # getting color:class name mapping
    color_map = {}
    for item in get_wordnet():
          color_map[str(item['color'])] = item['name']
    # print(f"color_map: {color_map}")
    # testing
    exact_color(lifted_mesh, color_mesh, scene_dir,np.array([0,0,0]))
    # end testing

    # For each unique color, extract and save the corresponding mesh
    # for i, color in enumerate(unique_colors):
    #     class_mesh = extract_class_vertices(lifted_mesh, color_mesh, color)
    #     if len(class_mesh.vertices) == 0:
    #         print(f"Skipping empty mesh for color {color}")
    #         continue  # Skip empty meshes
    #     try:
    #         class_name = color_map[str((255*color).astype(int).tolist())]
    #     except KeyError as e:
    #         print(f"KeyError caught: {e}")
    #         continue
    #     filename = f"{scene_dir}/single_{class_name}.ply"
    #     o3d.io.write_triangle_mesh(filename, class_mesh)
    #     print(f"Saved mesh for color {color} as {filename}")    
def match_similar_class(labels_3d):
    # swivel_chair->chair
    mask_swivel_chair = labels_3d == 18
    labels_3d[mask_swivel_chair] = 2
    # desk->table
    mask_desk = labels_3d == 17
    labels_3d[mask_desk] = 8
def get_single_class(
    scene_dir: Union[str, Path],
    maximum_label: int,
):
  # according to consensus label, get mesh of each class and save it
  scene_dir = Path(scene_dir)
  # define all paths
  input_color_dir = scene_dir / 'color'
  input_depth_dir = scene_dir / 'depth'
  input_intrinsic_dir = scene_dir / 'intrinsic'
  input_pose_dir = scene_dir / 'pose'
  input_label_dir = scene_dir / 'intermediate' / 'consensus'
  input_mesh_path = scene_dir / 'mesh.ply'
  log.info('Processing {} using for labels {}'.format(
      str(scene_dir),
      str(input_label_dir),
  ))
  # load mesh and extract colors
  mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))
  vertices = np.asarray(mesh.vertices)
  vertice_colors = np.asarray(mesh.vertex_colors)
  faces = np.asarray(mesh.triangles)
  # init label container
  labels_3d = np.zeros((vertices.shape[0], maximum_label + 1))
  files = input_label_dir.glob('*.png')
  files = sorted(files, key=lambda x: int(x.stem.split('.')[0]))
  resize_image = False
  for idx, file in tqdm(enumerate(files), total=len(files)):
    frame_key = file.stem
    intrinsics = np.loadtxt(str(input_intrinsic_dir / f'{frame_key}.txt'))
    image = np.asarray(Image.open(str(input_color_dir /
                                      f'{frame_key}.jpg'))).astype(np.uint8)
    depth = np.asarray(Image.open(str(
        input_depth_dir / f'{frame_key}.png'))).astype(np.float32) / 1000.
    labels = np.asarray(Image.open(str(file)))
    max_label = np.max(labels)
    if max_label > labels_3d.shape[-1] - 1:
      raise ValueError(
          f'Label {max_label} is not in the label range of {labels_3d.shape[-1]}'
      )
    if resize_image:
      h, w = depth.shape
      image = cv2.resize(image, (w, h))
      labels = cv2.resize(labels, (w, h))
    else:
      h, w, _ = image.shape
      depth = cv2.resize(depth, (w, h))
    pose_file = input_pose_dir / f'{frame_key}.txt'
    pose = np.loadtxt(str(pose_file))
    points_p = project_pointcloud(vertices, pose, intrinsics)
    xx = points_p[:, 0].astype(int)
    yy = points_p[:, 1].astype(int)
    zz = points_p[:, 2]
    valid_mask = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
    d = depth[yy[valid_mask], xx[valid_mask]]
    valid_mask[valid_mask] = (zz[valid_mask] > 0) & (np.abs(zz[valid_mask] - d)
                                                     <= 0.1)
    labels_2d = labels[yy[valid_mask], xx[valid_mask]]
    labels_3d[valid_mask, labels_2d] += 1
  second_column=labels_3d[:,1]
  # extract labels
  labels_3d = np.argmax(labels_3d, axis=-1)
  match_similar_class(labels_3d)
  unique_classes = np.unique(labels_3d)
  print(f"unique_classes: {unique_classes}")

  color_map = {}
  for item in get_wordnet():
    color_map[item['id']] = item['name']
  for class_id in tqdm(unique_classes, total=len(unique_classes)):
    class_name=color_map[class_id]
    class_mask = labels_3d == class_id
    indexes=np.where(second_column == class_id)[0]
    # print(f"indexes: {indexes}")
    class_vertices = vertices[class_mask]
    class_mesh = o3d.geometry.TriangleMesh()
    class_mesh.vertices = o3d.utility.Vector3dVector(class_vertices)
    class_mesh.vertex_colors = o3d.utility.Vector3dVector(vertice_colors[class_mask])
    new_faces = []   
    o3d.io.write_triangle_mesh(str(scene_dir /'single_class'/ f'{class_name}.ply'), class_mesh)
     
@gin.configurable
def main(
    scene_dir: Union[str, Path],
    label_folder: Union[str, Path],
    output_file: Union[str, Path],
    output_mesh: Union[str, Path],
    maximum_label: int,
):
  scene_dir = Path(scene_dir)
  label_folder = Path(label_folder)
  output_file = Path(output_file)
  output_mesh = Path(output_mesh)
  # check if scene_dir exists
  assert scene_dir.exists() and scene_dir.is_dir()
  # color_mesh_dir=scene_dir / 'mesh.ply'
  # lifted_mesh_dir=scene_dir / 'point_lifted_mesh.ply'
  # color_mesh = load_mesh(str(color_mesh_dir))
  # lifted_mesh = load_mesh(str(lifted_mesh_dir))
  # save_class_meshes(lifted_mesh, color_mesh, scene_dir)
  get_single_class(scene_dir,maximum_label)

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
  parser.add_argument('--label_folder', default='intermediate/consensus')
  parser.add_argument(
      '--max_label',
      type=int,
      default=200,
      help='Max label value',
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  main(
      scene_dir=args.workspace,
      label_folder=args.label_folder,
      output_file=args.output,
      output_mesh=args.output_mesh,
      maximum_label=args.max_label,
  )
