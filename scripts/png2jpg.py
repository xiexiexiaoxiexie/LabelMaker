import os
from PIL import Image

# Specify the folder containing PNG images
input_folder = '../azure/2024-11-02-17-00-25/color'  # Change this to your input folder
output_folder = '../azure/2024-11-02-17-00-25/color_'  # Change this to your output folder

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # Construct full file path
        png_file_path = os.path.join(input_folder, filename)
        
        # Open the PNG image
        with Image.open(png_file_path) as img:
            # Convert the image to RGB (necessary for JPEG)
            rgb_img = img.convert('RGB')
            
            # Define the output file path with .jpg extension
            jpg_file_path = os.path.join(output_folder, filename[:-4] + '.jpg')
            
            # Save the image as JPEG
            rgb_img.save(jpg_file_path, 'JPEG')
            print(f'Converted: {png_file_path} to {jpg_file_path}')

print('Conversion completed!')
