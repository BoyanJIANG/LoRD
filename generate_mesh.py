import torch
import os
import argparse
import time
import yaml
import trimesh
import pickle
from os.path import join
from lib.models.lord_model import LoRD
from lib.mesh_creator import create_mesh
from lib.models import decoder

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 4D model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--g', type=str, default='0', help='gpu id')
parser.add_argument('--seq_name', type=str, required=True, help='name of the sub-sequence')
parser.add_argument('--exp_name', type=str, default='fit_sparse_pcl', help='name of the experiment')
parser.add_argument('--pcl_type', type=str, choices=['pcl_test', 'depth_pcl'],
                        default='pcl_test', help='type of the observed point clouds')
parser.add_argument('--texture', action='store_true', help='whether to enable the texture model')
parser.add_argument('--use_h4d_smpl', action='store_true', help='whether to use the SMPL fitted by H4D')
parser.add_argument('--grid_size', type=float, default=0.007, help='size of the mesh extracting grid')

args = parser.parse_args()

with open(join('configs', args.config + '.yaml'), 'r') as f:
    cfg = yaml.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = args.g
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']

# Model
model = LoRD(in_dim=cfg['model']['in_dim'],
             motion_feat_dim=cfg['model']['latent_dim']).to(device)
if args.texture:
    tex_model = decoder.TextureField(c_dim=cfg['model']['latent_dim'], dim=cfg['model']['in_dim']).to(device)
else:
    tex_model = None


params_dict = torch.load(
    join(out_dir, 'checkpoints', 'model.pt')
)
print('=> Loading checkpoint from local file...')
model.load_state_dict(params_dict['model'])


radius = cfg['training']['part_size']
exp_name = args.exp_name
pcl_type = args.pcl_type

# you can append more sequences to the list
seqs = [
    args.seq_name
]

for p in seqs:
    human_id = '_'.join(p.split('_')[:-1])
    start_idx = int(p.split('_')[-1])
    model_name = f'{human_id}_{start_idx}'

    # Output directory
    gen_dir = join(out_dir, exp_name, 'vis')
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)

    load_dict = torch.load(join(out_dir, exp_name, 'latent_vecs', f'{model_name}.pt'))
    if args.texture:
        tex_model.load_state_dict(load_dict['tex_model'])
        tex_model.eval()

    tex_vec = load_dict['c_t'].to(device) if args.texture else None
    latent_vec = torch.cat([load_dict['c_m'], load_dict['c_s']], -1).to(device)
    model.eval()

    smpl_mesh_path = os.path.join('dataset', model_name, 'h4d_fitting' if args.use_h4d_smpl else 'body_mesh')
    pcl_path = os.path.join('dataset', model_name, pcl_type)

    crop_size = args.grid_size

    save_path = join(gen_dir, model_name, str(crop_size))
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for time_step in range(cfg['data']['length_sequence']):
            smpl_mesh = trimesh.load(os.path.join(smpl_mesh_path, f'{time_step:04d}.ply'), process=False)
            bbox = smpl_mesh.bounds
            create_mesh(load_dict, model, tex_model, latent_vec, tex_vec, time_step,
                        join(save_path, f'{time_step:04d}'), radius=radius,
                        crop_size=crop_size, bbox=bbox,
                        max_batch=20000, pcl_path=pcl_path)