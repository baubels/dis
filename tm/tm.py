
import time
import bpy
import numpy as np
from mathutils import Euler
from starfish import Sequence
from starfish.utils import random_rotations
from starfish.annotation import normalize_mask_colors, get_centroids_from_mask, get_bounding_boxes_from_mask
import sys
sys.path.append("/Users/dk/Documents.nosync/msc-project/blender/blender-gens")


# all satellites:
satellite = 'starlink' # one of 'starlink', 'hubble', 'cuerpo', 'xyz'
num_renders = 500
suns_per_template = 1

# WHICH SATELLITE
#
scene_name = 'Scene' if satellite in ['cuerpo', 'hubble', 'xyz'] else 'Real'
sat_name = 'Hubble' if satellite in ['hubble'] else 'SAT01Cuerpo' if satellite in ['cuerpo'] else 'Sat'
cam_name = 'Camera'
sun_name = 'Light' if satellite in ['cuerpo', 'hubble'] else 'Sun'
####################

# WHICH DATASET
#

if satellite == 'starlink': distances = [10]   # [10, 30]
elif satellite == 'xyz':    distances = [20]   # [20, 40]
elif satellite == 'hubble': distances = [40]   # [40, 60]
elif satellite == 'cuerpo': distances = [50]   # [50, 70]

attitudes = random_rotations(num_renders)
attitudes *= suns_per_template

seq1 = Sequence.standard(
    pose=attitudes,
    # lighting=random_rotations(num_renders*suns_per_template),
    distance=distances)
save_path = f"/Users/dk/Documents.nosync/msc-project/datasets/filtering/tm_{satellite}_distance_{distances[0]}_count_{num_renders*suns_per_template}/data"

# ------ this part is to be unchanged ------

import os
if not os.path.exists(save_path): os.makedirs(save_path)

# save this python file ran to the save path
import shutil
shutil.copy(__file__, os.path.join(os.path.dirname(save_path)))

# generate keypoints
for seq in [seq1]:
    # render loop
    for i, frame in enumerate(seq):

        # non-starfish Blender stuff: e.g. setting file output paths
        bpy.context.scene.render.resolution_x             = 224
        bpy.context.scene.render.resolution_y             = 224
        bpy.context.scene.render.use_high_quality_normals = True
        bpy.context.scene.render.use_stamp                = False
        
        if satellite == 'hubble':
            # set light to be as if it's the sun
            lamp = bpy.data.objects['Light']
            lamp.data.energy = 5  # 10 is the max value for energy
            lamp.data.type = 'SUN'  # in ['POINT', 'SUN', 'SPOT', 'HEMI', 'AREA']
            lamp.data.distance = 50
        else: lamp = 0

        sun_name_used = bpy.data.objects[sun_name] if satellite != 'hubble' else lamp
        # set up and render
        scene = bpy.data.scenes[scene_name]
        frame.setup(scene, 
                    bpy.data.objects[sat_name],
                    bpy.data.objects[cam_name],
                    sun_name_used
                    )
        bpy.context.scene.render.filepath = f'{save_path}/real_{i}.png'
        bpy.ops.render.render(write_still=True)

        # add some extra metadata
        frame.timestamp = int(time.time() * 1000)
        frame.attitude = attitudes[i]

        # save metadata to JSON
        with open(f'{save_path}/meta_{i}.json', 'w') as f:
            f.write(frame.dumps())
