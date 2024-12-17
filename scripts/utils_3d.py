import argparse
import os
from os.path import exists, join

import cv2
import gin
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm


@gin.configurable
def fuse_mesh(
    scan_dir: str,
    sdf_trunc: float = 0.06,
    voxel_length: float = 0.02,
    depth_trunc: float = 6.0,
    depth_scale: float = 1000.0,
):

  color_dir = join(scan_dir, 'color')
  depth_dir = join(scan_dir, 'depth')
  pose_dir = join(scan_dir, 'pose')
  intrinsic_dir = join(scan_dir, 'intrinsic')

  assert exists(color_dir)
  assert exists(depth_dir)
  assert exists(pose_dir)
  assert exists(intrinsic_dir)

  color_list = os.listdir(color_dir)
  color_list.sort(key=lambda e: int(e[:-4]))

  depth_list = os.listdir(depth_dir)
  depth_list.sort(key=lambda e: int(e[:-4]))

  pose_list = os.listdir(pose_dir)
  pose_list.sort(key=lambda e: int(e[:-4]))

  intr_list = os.listdir(intrinsic_dir)
  intr_list.sort(key=lambda e: int(e[:-4]))

  # see if all files exists
  assert all(
      (a[:-4] == b[:-4]) and (a[:-4] == c[:-4]) and (a[:-4] == d[:-4])
      for a, b, c, d in zip(color_list, depth_list, pose_list, intr_list))

  tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
      sdf_trunc=sdf_trunc,
      voxel_length=voxel_length,
      color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

  for color_f, depth_f, pose_f, intr_f in tqdm(
      zip(color_list, depth_list, pose_list, intr_list),
      total=len(color_list),
  ):

    intr = np.loadtxt(join(intrinsic_dir, intr_f))
    pose = np.loadtxt(join(pose_dir, pose_f))
    color = np.asanyarray(Image.open(join(color_dir, color_f))).astype(np.uint8)
    depth = np.asarray(Image.open(join(depth_dir, depth_f))).astype(np.uint16)

    # fujing resize the image/depth/intrinsics
    # resize_ratio=3.2 # 1536/480
    # color = cv2.resize(color, (int(color.shape[1] / resize_ratio), int(color.shape[0] / resize_ratio)))
    # depth = cv2.resize(depth, (int(depth.shape[1] / resize_ratio), int(depth.shape[0] / resize_ratio)))
    # intr[0, 0] /= resize_ratio
    # intr[1, 1] /= resize_ratio
    # intr[0, 2] /= resize_ratio
    # intr[1, 2] /= resize_ratio


    h, w, _ = color.shape
    color = o3d.geometry.Image(color)
    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color,
        depth=depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False)
    # intrinsics = o3d.camera.PinholeCameraIntrinsic(
    # width=w,
    # height=h,
    # fx=intr[0, 0],
    # fy=intr[1, 1],
    # cx=intr[0, 2],
    # cy=intr[1, 2]
    # )
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    # rgbd,
    # intrinsics
    # )
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="3D RGBD Visualization")
    # vis.add_geometry(pcd)
    # vis.run()

    camera_to_pose = np.array([
    [-1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
    ])
    # pose_=np.dot(pose,camera_to_pose)
    pose_=pose

    tsdf.integrate(
        image=rgbd,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            height=h,
            width=w,
            fx=intr[0, 0],
            fy=intr[1, 1],
            cx=intr[0, 2],
            cy=intr[1, 2]
        ),
        extrinsic=np.linalg.inv(pose_),
    )
  
  mesh = tsdf.extract_triangle_mesh()
  o3d.io.write_triangle_mesh(join(scan_dir, 'mesh.ply'), mesh)

def simplify_mesh(scan_dir,mesh, target_faces):
  simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=800000)
  o3d.io.write_triangle_mesh(join(scan_dir, 'simplified_mesh.ply'), simplified_mesh)


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--workspace", type=str)
  parser.add_argument("--sdf_trunc", type=float, default=0.04)
  parser.add_argument("--voxel_length", type=float, default=0.008)
  parser.add_argument("--depth_trunc", type=float, default=3.0)
  parser.add_argument("--depth_scale", type=float, default=1000.0)
  parser.add_argument('--config', help='Name of config file')

  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  fuse_mesh(
      scan_dir=args.workspace,
      sdf_trunc=args.sdf_trunc,
      voxel_length=args.voxel_length,
      depth_trunc=args.depth_trunc,
      depth_scale=args.depth_scale,
  )
  simplify_mesh(args.workspace, o3d.io.read_triangle_mesh(args.workspace+'/mesh_original.ply'), 20000)
  
