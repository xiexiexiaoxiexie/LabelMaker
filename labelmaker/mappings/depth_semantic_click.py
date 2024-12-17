import argparse
import logging
from pathlib import Path
from typing import Union
import cv2
import gin
from labelmaker.label_data import get_wordnet
from labelmaker.label_data import get_ade150
from labelmaker.label_data import get_nyu40
from labelmaker.label_data import get_scannet200
import numpy as np

# python labelmaker/mappings/mesh_screenshot_click.py --workspace $WORKSPACE_DIR --depth_number 0

logging.basicConfig(level="INFO")
log = logging.getLogger('Get class name from consensued depth image by clicking')
def find_non_zero_pixels(depth_image):
    # Get the coordinates of non-zero pixels
    non_zero_coords = np.column_stack(np.where(depth_image != 0))
    
    # Output the coordinates
    if non_zero_coords.size > 0:
        print("Non-zero pixels (x, y):")
        for coord in non_zero_coords:
            print(f"({coord[1]}, {coord[0]})")  # (x, y) -> (column, row)
    else:
        print("No non-zero pixels found.")
    
    return non_zero_coords
def get_class_name(event, x, y, flags, param):
    depth_image, model_name = param
    # print('depth_image: ', depth_image.shape)
    # find_non_zero_pixels(depth_image)
    model_prefix = ''
    if model_name == 'consensus':
        model_prefix = 'wordnet'
    else:
        model_prefix= model_name.split('_')[0]
    if model_prefix=='ade20k':
        model_prefix='ade150'
    if event == cv2.EVENT_LBUTTONDOWN:
        # Retrieve the depth value 
        original_value = depth_image[y, x]
        color_map = {}
        for item in globals()['get_'+model_prefix]():
          color_map[item['id']] = item['name']
        # Get the class name from the dictionary, if available
        print(f"Depth value: {original_value}")
        print(f'x: {x}, y: {y}')
        print(f"Class name: {color_map[original_value]}")
        return original_value
    
@gin.configurable
def main(
    scene_dir: Union[str, Path],
    label_folder: Union[str, Path],
    output_file: Union[str, Path],
    output_mesh: Union[str, Path],
    maximum_label: int,
    depth_number: str,
    model_name: str,
):
  scene_dir = Path(scene_dir)
  label_folder = Path(label_folder)
  output_file = Path(output_file)
  output_mesh = Path(output_mesh)

  assert scene_dir.exists() and scene_dir.is_dir()
  # define all paths
  input_color_dir = scene_dir / 'color'/f"{str(depth_number)}.jpg"
  input_depth_dir = scene_dir
  input_depth_dir = input_depth_dir / 'intermediate' / model_name /f"{str(depth_number)}.png"
  depth_image = cv2.imread(input_depth_dir, cv2.IMREAD_UNCHANGED)
  unique_values = np.unique(depth_image)
  print(unique_values)
  color_image = cv2.imread(input_color_dir, cv2.IMREAD_COLOR)
  depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

  cv2.imshow('Color image', color_image)
  cv2.setMouseCallback('Color image', get_class_name, param=(depth_image,model_name))
  print("Click on the image to get the class name for that pixel.")
  cv2.waitKey(0) 
  cv2.destroyAllWindows()



def arg_parser():
  parser = argparse.ArgumentParser(
      description=
      'Project 3D points to 2D image plane and aggregate labels and save label txt'
  )
  parser.add_argument(
      '--depth_number',
      type=str,
      required=True,
      help=
      'depth or color image number in consensus folder',
  )
  parser.add_argument(
      '--model_name',
      type=str,
      required=True,
      help=
      'The model name',
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
      default=2000,
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
      depth_number=args.depth_number,
      model_name=args.model_name,
  )
