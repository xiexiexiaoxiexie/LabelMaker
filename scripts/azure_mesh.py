import argparse
import json
import logging
import os
import shutil
import sys
from os.path import abspath, dirname, exists, join

import cv2
import gin
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import trange

sys.path.append(abspath(join(dirname(__file__), '..')))
from utils_3d import fuse_mesh
from utils_3d import simplify_mesh
import open3d as o3d


@gin.configurable
def process_arkit():

  # scan_dir="/home/xiefujing/LabelMaker/azure/2024-11-02-17-00-25"
  target_dir="/home/xiefujing/LabelMaker/azure/2024-11-03-09-47-53"
  sdf_trunc=0.04
  voxel_length=0.008
  depth_trunc=3.0
  depth_scale=1000.0
  # fuse_mesh(
  #     scan_dir=target_dir,
  #     sdf_trunc=sdf_trunc,
  #     voxel_length=voxel_length,
  #     depth_trunc=depth_trunc,
  #     depth_scale=depth_scale,
  # )  # depth_scale is a fixed value in ARKitScene, no need to pass an argument in cli
  mesh_in = o3d.io.read_triangle_mesh(target_dir+'/mesh_1536.ply')
  simplify_mesh(target_dir,mesh_in, 20000)




if __name__ == "__main__":

  process_arkit()
