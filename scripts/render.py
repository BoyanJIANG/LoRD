### This script is adapted from: https://github.com/shunsukesaito/SCANimate/blob/main/render/render_aist.py
### basic usage: python render.py -i <mesh folder>


import open3d as o3d
import numpy as np

import os
from os.path import join

import argparse

from tqdm import tqdm
import random

import cv2


def render_single_image(result_mesh_file, output_image_file, vis, yprs, raw_color=False,
                        normalization=True, scale=None, center=None, floor=None):
    # Render result image
    mesh = o3d.io.read_triangle_mesh(result_mesh_file)
    mesh.compute_vertex_normals()
    if not raw_color:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

    verts = np.asarray(mesh.vertices)

    if normalization:
        if scale is None:
            bbox = [verts.min(0), verts.max(0)]
            scale = (bbox[1] - bbox[0]).max()
            center = (bbox[0] + bbox[1]) / 2

        if floor is not None:
            verts += np.array([0.0, floor - verts.min(0)[1], 0.0])

        verts = (verts - center) / scale
        mesh.vertices = o3d.utility.Vector3dVector(verts)

    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    for ypr in yprs:
        ctr.rotate(0, RENDER_RESOLUTION / 180 * ypr[1])
        ctr.rotate(RENDER_RESOLUTION / 180 * ypr[0], 0)

    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(False)
    result_image = vis.capture_screen_float_buffer(False)
    vis.clear_geometries()

    result_img = np.asarray(result_image)
    depth_img = np.asarray(depth)
    depth_img = (depth_img * 1000).astype(np.uint16)  # resolution 1mm

    cv2.imwrite(output_image_file.replace('.png', '_rgb.png'), result_img[:, :, ::-1] * 255.)
    cv2.imwrite(output_image_file, depth_img)

    return result_img, scale, center


random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default=None, help='result directory')
parser.add_argument('-txt', '--input_dir_list', type=str, default=None, help='result directory')
parser.add_argument('-o', '--out_dir', type=str, default='', help='Output directory or filename')

parser.add_argument('-color', '--has_color', action='store_true', help='if the input meshes contain vertices color')
parser.add_argument('-postfix', '--postfix', type=str, default='', help='assign postfix of mesh files')

parser.add_argument('-smooth', '--smooth_shade', action='store_true', help='weather to use smooth shading')
parser.add_argument('-normal', '--normal', action='store_true', help='render normal map')
parser.add_argument('-black_bg', '--background', action='store_true', help='default white background')

parser.add_argument('-azimuth', '--azimuth', type=int, default=0, help='camera azimuth')
parser.add_argument('-elevation', '--elevation', type=int, default=0, help='camera elevation')

parser.add_argument('-norm', '--normalization', type=str, default='none',
                    choices=['none', 'independent', 'global', 'align_floor'], help='normalization type')

args = parser.parse_args()
print(args)

if not args.input_dir_list:
    input_dirs = [args.input_dir]
else:
    with open(args.input_dir_list, 'r') as f:
        input_dirs = f.read().split('\n')
    input_dirs = list(filter(lambda x: len(x) > 0, input_dirs))


vis = o3d.visualization.Visualizer()
RENDER_RESOLUTION = 512
vis.create_window(width=RENDER_RESOLUTION, height=RENDER_RESOLUTION)

render_option = "scripts/option.json"

print(f'Loading render option: {render_option}')
vis.get_render_option().load_from_json(render_option)

opt = vis.get_render_option()
if args.background:
    opt.background_color = np.asarray([0, 0, 0])
if args.normal:
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
if args.smooth_shade:
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

for dir_index in range(len(input_dirs)):
    print(f'dir_index: {dir_index}')
    args.input_dir = input_dirs[dir_index]

    input_dir = args.input_dir
    output_dir = args.out_dir
    if output_dir == '':
        output_dir = input_dir
    output_dir = output_dir[:-1] if output_dir[-1] == '/' else output_dir

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    INTRINSIC = np.eye(3, dtype=np.float32)
    INTRINSIC[0, 0] = 1 / (32 / 35 / RENDER_RESOLUTION)
    INTRINSIC[1, 1] = 1 / (32 / 35 / RENDER_RESOLUTION)
    INTRINSIC[0, 2] = RENDER_RESOLUTION / 2 - 0.5
    INTRINSIC[1, 2] = RENDER_RESOLUTION / 2 - 0.5
    cam_intrinsics.intrinsic_matrix = INTRINSIC

    cam_intrinsics.width = RENDER_RESOLUTION
    cam_intrinsics.height = RENDER_RESOLUTION

    EXTRINSIC = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, 3.5],
                          [0.0, 0.0, 0.0, 1.0]])
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = cam_intrinsics
    cam_params.extrinsic = EXTRINSIC

    output_folder = join(output_dir, 'rendering')
    os.makedirs(output_folder, exist_ok=True)

    has_color = args.has_color

    mesh_files = sorted([f for f in os.listdir(input_dir) if f'{args.postfix}.obj' in f or f'{args.postfix}.ply' in f])

    # control the azimuth and elevation of camera
    yprs = [[args.azimuth, args.elevation]]

    if args.normalization in ['none', 'independent']:
        for i, mesh_file in enumerate(tqdm(mesh_files[::1])):
            mesh_file_path = join(input_dir, mesh_file)
            output_image_file = join(output_folder, str(i).zfill(4) + '.png')
            _ = render_single_image(mesh_file_path, output_image_file, vis, yprs, has_color,
                                    normalization=False if args.normalization == 'none' else True)
    else:
        mesh = o3d.io.read_triangle_mesh(join(input_dir, mesh_files[0]))
        verts = np.asarray(mesh.vertices)
        bbox = [verts.min(0), verts.max(0)]
        scale0 = (bbox[1] - bbox[0]).max()
        center0 = (bbox[0] + bbox[1]) / 2
        floor = verts.min(0)[1] if args.normalization == 'align_floor' else None
        for i, mesh_file in enumerate(tqdm(mesh_files[::1])):
            mesh_file_path = join(input_dir, mesh_file)
            output_image_file = join(output_folder, str(i).zfill(4) + '.png')
            _ = render_single_image(mesh_file_path, output_image_file, vis, yprs, has_color,
                                    scale=scale0, center=center0, floor=floor)


