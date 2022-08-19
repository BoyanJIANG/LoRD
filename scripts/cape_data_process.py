import os
import numpy as np
import trimesh
import pickle
import torch
from smplx import SMPL
from tqdm import tqdm
from trimesh.sample import sample_surface
from lib.utils.point_utils import write_point_ply, sample_mesh


SEQ_LEN = 17
SAMPLES = 250000
DATA_PATH = 'data/demo'
OUT_PATH = 'dataset'

# control the density of the sampled point cloud
pts_per_m2 = 2000

smpl_faces = np.load('data/smpl_faces.npy')
smpl_betas = pickle.load(open('data/shapes.pkl', 'rb'))
body_model = SMPL('data/SMPL_NEUTRAL.pkl')


seqs = [
    '03284_shortlong_simple_87',
    '00134_longlong_twist_trial2_21',
    '02474_longshort_ROM_lower_258'
]

for seq in tqdm(seqs):
    seq_sp = seq.split('_')
    sid = seq_sp[0]
    mid = '_'.join(seq_sp[1:-1])
    start_idx = int(seq_sp[-1])

    # GT mesh
    mesh_out_folder = os.path.join(OUT_PATH, seq, 'mesh')
    os.makedirs(mesh_out_folder, exist_ok=True)

    # SMPL body mesh
    smpl_mesh_out_folder = os.path.join(OUT_PATH, seq, 'body_mesh')
    os.makedirs(smpl_mesh_out_folder, exist_ok=True)

    # dense point clouds for training
    train_pcl_out_folder = os.path.join(OUT_PATH, seq, 'pcl_train')
    os.makedirs(train_pcl_out_folder, exist_ok=True)

    # sparse point clouds for testing
    test_pcl_out_folder = os.path.join(OUT_PATH, seq, 'pcl_test')
    os.makedirs(test_pcl_out_folder, exist_ok=True)

    for i in range(SEQ_LEN):
        file = os.path.join(DATA_PATH, seq, 'raw', f'{mid}.{(start_idx+i):06d}.npz')
        frame_data = np.load(file)

        # SMPL body mesh
        pose = torch.Tensor(frame_data['pose'])[None]
        beta = torch.Tensor(smpl_betas[sid])[None]
        smpl_verts = body_model(
            betas=beta,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3]
        ).vertices.detach().cpu().numpy()[0]
        body_mesh = trimesh.Trimesh(smpl_verts, smpl_faces, process=False)
        body_mesh.export(os.path.join(smpl_mesh_out_folder, f'{i:04d}.ply'))

        # clothed mesh
        full_verts = frame_data['v_posed'] - frame_data['transl']
        full_mesh = trimesh.Trimesh(full_verts, smpl_faces, process=False)
        full_mesh.export(os.path.join(mesh_out_folder, f'{i:04d}.ply'))

        # points for training
        sample = sample_surface(full_mesh, SAMPLES)
        pnts = sample[0]
        normals = full_mesh.face_normals[sample[1]]
        point_set = np.hstack([pnts, normals])
        np.save(os.path.join(train_pcl_out_folder, f'{i:04d}.npy'), point_set)

        # points for testing
        sample_pts, sample_normals = sample_mesh(full_mesh, 1. / pts_per_m2)
        write_point_ply(os.path.join(test_pcl_out_folder, f'{i:04d}.ply'), sample_pts, sample_normals)
