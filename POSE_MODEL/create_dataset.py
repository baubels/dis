import os
from PIL import Image
from tqdm import tqdm
import shutil


# Define the input and output directories and the size of the new images
input_dir =  "/Users/dk/Documents.nosync/msc-project/blender/imgs/starlink-high-res/2k/closer/standard"
output_dir = "/Users/dk/Documents.nosync/msc-project/blender/imgs/starlink-high-res/2k/closer/dataset"

if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

new_size = (224, 224)

images = [img for img in os.listdir(input_dir) if img.endswith(".png")]
images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
descriptions = [desc for desc in os.listdir(input_dir) if desc.endswith(".json")]
descriptions.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


for i in tqdm(range(len(images))):
    # resize and copy over the images
    image = Image.open(os.path.join(input_dir, images[i]))
    resized_image = image.resize(new_size)
    new_filename = 'Sat_small_' + images[i].split('_')[1]
    resized_image.save(os.path.join(output_dir, new_filename))
    
    # copy over the .json file
    input_path = os.path.join(input_dir, f'meta_{i}.json')
    output_path = os.path.join(output_dir, f'meta_{i}.json')
    shutil.copyfile(input_path, output_path)
