from PIL import Image
import os

def convert_images_to_grayscale(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', 'webp')):
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                grayscale_img = img.convert("L")
                grayscale_img.save(os.path.join(output_folder, filename))

    print(f"All images from '{input_folder}' have been converted to grayscale and saved in '{output_folder}'.")

input_folder_path = '/home/omen/Downloads/img'
output_folder_path = '/home/omen/Downloads/gray_img'

convert_images_to_grayscale(input_folder_path, output_folder_path)
