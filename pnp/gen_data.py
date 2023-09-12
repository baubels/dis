
import time
import bpy
import numpy as np
from mathutils import Euler
from starfish import Sequence
from starfish.utils import random_rotations
from starfish.annotation import normalize_mask_colors, get_centroids_from_mask, get_bounding_boxes_from_mask
import sys
sys.path.append("/Users/dk/Documents.nosync/msc-project/blender/blender-gens")



# WHICH SATELLITE
satellite = 'hubble' # one of 'starlink', 'hubble', 'cuerpo', 'xyz'
#
scene_name = 'Scene' if satellite in ['cuerpo', 'hubble'] else ['Real']
sat_name = 'Hubble' if satellite in ['hubble'] else 'SAT01Cuerpo' if satellite in ['cuerpo'] else 'Sat'
cam_name = 'Camera'
sun_name = 'Light' if satellite in ['cuerpo', 'hubble'] else 'Sun'
num_keypoints = 11
####################

# WHICH DATASET
dataset = 1
#
num = 100
attitudes = np.linspace([0,0,0], [0, 1.5*np.pi, 1.5*np.pi], num=num) if dataset==1 else np.linspace([np.pi,0,0], [0, 1.5*np.pi, 1.5*np.pi], num=num) if dataset == 2 else np.linspace([np.pi,0,-np.pi], [0, 1.5*np.pi, 1.5*np.pi], num=num)
attitudes = attitudes.tolist()
attitudes_euler = [Euler(attitude) for attitude in attitudes]
offset = [(0.2, 0.2), (0.4, 0.4)] if dataset == 1 else [(0.5, 0.5)]
distances = [10,30] if satellite in ['starlink', 'hubble'] and dataset==1 else [20,40]
seq1 = Sequence.standard(
    pose=attitudes_euler,
    offset=offset,
    distance=distances)
save_path = f"/Users/dk/Documents.nosync/msc-project/datasets/filtering/{satellite}/{dataset}/data"

# ## DATASET 2
# num = 100
# attitudes = np.linspace([np.pi,0,0], [0, 1.5*np.pi, 1.5*np.pi], num=num)
# attitudes = attitudes.tolist()
# attitudes_euler = [Euler(attitude) for attitude in attitudes]
# offset = [(0.5, 0.5)]
# distances = [20,40]
# seq1 = Sequence.standard(
#     pose=attitudes_euler,
#     offset=offset,
#     distance=distances)
# save_path = f"/Users/dk/Documents.nosync/msc-project/datasets/filtering/{satellite}/{dataset}/data"

# ## DATASET 3
# num = 100
# attitudes = np.linspace([np.pi,0,-np.pi], [0, 1.5*np.pi, 1.5*np.pi], num=num)
# attitudes = attitudes.tolist()
# attitudes_euler = [Euler(attitude) for attitude in attitudes]
# offset = [(0.5, 0.5)]
# distances = [40,40]
# seq1 = Sequence.standard(
#     pose=attitudes_euler,
#     offset=offset,
#     distance=distances)
# save_path = f"/Users/dk/Documents.nosync/msc-project/datasets/filtering/{satellite}/{dataset}/data"


# ------ this part is to be unchanged ------


import os
if not os.path.exists(save_path):
    os.makedirs(save_path)


# save this python file ran to the save path
import shutil
shutil.copy(__file__, os.path.join(os.path.dirname(save_path)))


from starfish.annotation.generate_keypoints import generate_keypoints
from starfish.annotation.keypoints import project_keypoints_onto_image



scene_base = bpy.data.scenes[scene_name]
sat_base = bpy.data.objects[sat_name]
cam_base = bpy.data.objects[cam_name]
world_keypoints = generate_keypoints(scene_base.objects[sat_name], num_keypoints)


def get_calibration_matrix_K_from_blender(scene, camd):                    # GET INTRINSIC MATRIX
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    # camd = bpy.data.objects['Camera']
    f_in_mm = camd.lens
    # scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0                                            # only use rectangular pixels

    K = np.array([[alpha_u, skew,    u_0],
                  [    0  , alpha_v, v_0],
                  [    0  , 0,        1  ]])
    return K


for seq in [seq1]:
    # render loop
    for i, frame in enumerate(seq):

        # non-starfish Blender stuff: e.g. setting file output paths
        bpy.context.scene.render.resolution_x             = 224
        bpy.context.scene.render.resolution_y             = 224
        bpy.context.scene.render.use_high_quality_normals = True
        bpy.context.scene.render.use_stamp                = False

        # set up and render
        scene = bpy.data.scenes[scene_name]
        frame.setup(scene, 
                    bpy.data.objects[sat_name],
                    bpy.data.objects[cam_name], 
                    bpy.data.objects[sun_name])
        bpy.context.scene.render.filepath = f'{save_path}/real_{i}.png'
        bpy.ops.render.render(write_still=True)

        # add keypoints
        sensor_keypoints = project_keypoints_onto_image(world_keypoints, scene, bpy.data.objects[sat_name], bpy.data.objects[cam_name])
        frame.sensor_keypoints        = sensor_keypoints    # 2d keypoints as observed in the image
        # frame.nopose_sensor_keypoints = project_keypoints_onto_image(world_keypoints, scene_base, sat_base, cam_base)
        frame.world_keypoints         = world_keypoints     # ground truth 3d keypoints

        # add some extra metadata
        frame.timestamp = int(time.time() * 1000)
        frame.sequence_name = f'{num} random poses'
        # frame.tags = ['front_view', 'left_view', 'right_view']
        frame.attitude = attitudes[i]
        # frame.attitude_rot = attitudes_rot[i]

        # save intrinsic matrix to JSON
        K = get_calibration_matrix_K_from_blender(scene, bpy.data.objects[cam_name].data).tolist()
        frame.intrinsic_matrix = K

        # save metadata to JSON
        with open(f'{save_path}/meta_{i}.json', 'w') as f:
            f.write(frame.dumps())
