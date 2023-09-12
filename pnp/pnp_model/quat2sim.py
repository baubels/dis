#!BPY
"""
Name: 'Take a sequence of pictures'
Blender: 249
Group: 'Add'
Tooltip: 'Takes a sequence of pictures of an object between two different viewpoints.'
"""

# import mathutils
import sys
sys.path.append("/Users/dk/Documents.nosync/msc-project/blender/scripts-main")


def gen_one(qauternion, item_idx, dir):
    # qauternion = mathutils.Quaternion(qauternion)

    # use item_idx to generate non-pose data
    import json
    import os

    label_file = os.path.join(dir, f'meta_{item_idx}.json')
    
    # generate the image from estimated pose
    import subprocess
    command = ['Blender', '--background', 
               '/Users/dk/Documents.nosync/msc-project/blender/models/starlink.blend',
               '--python', '/Users/dk/Documents.nosync/msc-project/blender/scripts-experimental/take_one_for_python.py', '--',
               '--sequence_type', 'none',
               '--label_file', f'{label_file}',
               '--resolution', '224', '224',
               '--pose_estimate', f'{qauternion[0]}',  f'{qauternion[1]}',  f'{qauternion[2]}',  f'{qauternion[3]}']
    try:
        output = subprocess.check_output(command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("\n\ncommand '{}' return with error (code {}): \n{}".format(e.cmd, e.returncode, e.output))
    
    rendered_img_filepath = output.decode('utf-8')
    
    # find the rendered image filepath
    import re
    match = re.search(r"!@£\$%(.+?)!@£\$%", rendered_img_filepath)

    if match:
        img_path = match.group(1)
    else: print("No match found")

    # return the rendered image
    import torchvision
    from torchvision.io import read_image
    image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
    return image
