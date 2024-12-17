import argparse
import logging
from pathlib import Path
from typing import Union
import gin
from labelmaker.label_data import get_wordnet
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk

# python labelmaker/mappings/mesh_screenshot_click.py --workspace $WORKSPACE_DIR --screenshot_dir screenshot

logging.basicConfig(level="INFO")
log = logging.getLogger('Get class name from lifted pointcloud screenshot by clicking')

# handling clicking on the canvas
def on_click(event,img,canvas,color_map):
    x = event.x
    y = event.y
    rgb_value = img.getpixel((x, y))
    color_key=[int(x) for x in rgb_value][0:3]
    if str(color_key) not in color_map:
      print(f"Color key not found: {color_key}")
      return
    # retrive class name from a dict of string color: class name
    color_class = color_map[str(color_key)]
    canvas.create_text(x, y, text=color_class, fill="black", font=('Helvetica'))

@gin.configurable
def main(
    scene_dir: Union[str, Path],
    label_folder: Union[str, Path],
    output_file: Union[str, Path],
    output_mesh: Union[str, Path],
    maximum_label: int,
    screenshot_dir: str,
):
  scene_dir = Path(scene_dir)
  label_folder = Path(label_folder)
  output_file = Path(output_file)
  output_mesh = Path(output_mesh)

  # check if scene_dir exists
  assert scene_dir.exists() and scene_dir.is_dir()
  input_color_dir = scene_dir /f"{str(screenshot_dir)}.png"
  img = Image.open(input_color_dir)

  # Create the main window
  root = tk.Tk()
  root.title("Click on screenshot, will return class name")
  tk_img = ImageTk.PhotoImage(img)
  # Create a canvas to display the image
  canvas = Canvas(root, width=tk_img.width(), height=tk_img.height())
  canvas.pack()
  canvas.create_image(0, 0, anchor="nw", image=tk_img)
  # create a dict of string color: class name
  color_map = {}
  for item in get_wordnet():
    color_map[str(item['color'])] = item['name']
  # Bind the mouse click event to the on_click function
  canvas.bind("<Button-1>", lambda event: on_click(event, img,canvas,color_map))
  root.mainloop()

def arg_parser():
  parser = argparse.ArgumentParser(
      description=
      'Project 3D points to 2D image plane and aggregate labels and save label txt'
  )
  parser.add_argument(
      '--screenshot_dir',
      type=str,
      help=
      'Path to the directory where the screenshots are stored.',
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
      screenshot_dir=args.screenshot_dir,
  )
