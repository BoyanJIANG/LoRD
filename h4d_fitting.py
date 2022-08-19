import os
import torch
import torch.optim as optim
import numpy as np
from os.path import join
import time
import argparse
from torch.nn import functional as F
from smplx import SMPL
import trimesh
import datetime
from trimesh.exchange.export import export_mesh
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from lib.utils import SmoothedValue
from lib.utils.h4d_utils import Prior, get_loss_weights, backward_step, compute_p2s_loss

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance



def pca_layer(bm, c_i, c_p, c_m, n_steps):
    pca = np.load('data/pca.npz')
    body_pca = torch.Tensor(pca['body_comp'][:256]).cuda()
    body_mean = torch.Tensor(pca['body_mean']).cuda()
    global_rot_pca = torch.Tensor(pca['global_comp'][:16]).cuda()
    global_rot_mean = torch.Tensor(pca['global_mean']).cuda()

    c_i_batch = c_i.repeat(n_steps, 1)

    delta_root_orient = torch.matmul(c_m[:, :16], global_rot_pca) + global_rot_mean
    delta_root_orient = delta_root_orient.view(n_steps-1, -1)
    delta_body_pose = torch.matmul(c_m[:, 16:], body_pca) + body_mean
    delta_body_pose = delta_body_pose.view(n_steps-1, -1)

    delta_pose = torch.cat([delta_root_orient, delta_body_pose], -1)
    poses = c_p + delta_pose  # 16, 72
    poses_stage1 = torch.cat([c_p, poses], 0)  # 17, 72

    pred_verts = bm(
        betas=c_i_batch,
        body_pose=poses_stage1[:, 3:],
        global_orient=poses_stage1[:, :3],
    ).vertices

    return pred_verts, poses_stage1


def back_optim(mesh_dir, device, code_std=0.01, num_iterations=500):
    id_code = torch.ones(1, 10).normal_(mean=0, std=code_std).cuda()
    pose_code = torch.ones(1, 72).normal_(mean=0, std=code_std).cuda()
    motion_code = torch.ones(1, 256+16).normal_(mean=0, std=code_std).cuda()

    id_code.requires_grad = True
    pose_code.requires_grad = True
    motion_code.requires_grad = True

    optimizer = optim.Adam([id_code, pose_code, motion_code], lr=0.03)
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    bm_path = 'data/SMPL_NEUTRAL.pkl'
    bm = SMPL(model_path=bm_path).to(device)
    faces = bm.faces

    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    gt_pcl = []
    for i in range(seq_len):
        pcl_file = join(gt_path, f'{i:04d}.ply')
        pcl = trimesh.load(pcl_file).vertices
        gt_pcl.append(torch.Tensor(pcl))
    gt_pcl = Pointclouds(points=gt_pcl).to(device)

    prior = Prior(sm=None)['Generic']
    weight_dict = get_loss_weights()
    batch_time = SmoothedValue()

    end = time.time()
    for step in range(num_iterations):

        v1, pred_pose = pca_layer(bm, id_code, pose_code, motion_code, n_steps=seq_len)

        faces_tensor = torch.from_numpy(faces.astype(np.int32)).to(device)

        pred_mesh = Meshes(verts=[v1[i] for i in range(seq_len)],
                           faces=[faces_tensor for _ in range(seq_len)])

        loss_dict = dict()

        if loss_type == 'p2s':
            loss_dict['point2surface'] = compute_p2s_loss(pred_mesh, gt_pcl, seq_len=seq_len)
        else:
            pred_pcl = sample_points_from_meshes(pred_mesh, 4096)
            loss_dict['chamfer'] = chamfer_distance(pred_pcl, gt_pcl)[0] / 2
        loss_dict['betas'] = torch.mean(id_code ** 2)
        loss_dict['pose_pr'] = prior(pred_pose).mean()

        tot_loss = backward_step(loss_dict, weight_dict, step)

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        lr_sche.step()

        t_batch = time.time() - end
        end = time.time()
        batch_time.update(t_batch)
        eta_seconds = batch_time.global_avg * (num_iterations - step + 1)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        summary_string = f'Step {step + 1} ({step + 1}/{num_iterations}) | ' \
                         f'ETA: {eta_string} | batch_time: {t_batch:.2f} | ' \
                         f'lr:{optimizer.param_groups[0]["lr"]:.6f}'

        for k, v in loss_dict.items():
            if 'point2surface' in k or 'chamfer' in k:
                summary_string += f' | {k}: {v:.6f}'
            else:
                summary_string += f' | {k}: {v:.4f}'

        print(summary_string)

        if (step + 1) % 100 == 0:
            # -----------visualization-------------
            for i, v in enumerate(v1):
                body_mesh = trimesh.Trimesh(vertices=c2c(v), faces=faces, process=False)
                export_mesh(body_mesh, os.path.join(mesh_dir, f'{i:04d}.ply'))

            # -----------save codes-------------
            print('Saving latent vectors...')
            torch.save(
                {"id_code": id_code.detach().cpu(),
                 "pose_code": pose_code.detach().cpu(),
                 "motion_code": motion_code.detach().cpu()},
                os.path.join(out_dir, 'latent_vecs.pt')
            )

            np.savez(os.path.join(out_dir, 'smpl_params.npz'),
                     beta=id_code.detach().cpu().numpy(),
                     poses=pred_pose.detach().cpu().numpy())


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 4D model.'
    )
    parser.add_argument('--g', type=str, default='0', help='gpu id')
    parser.add_argument('--seq_name', type=str, required=True, help='name of the sub-sequence')
    parser.add_argument('--pcl_type', type=str, choices=['pcl_test', 'depth_pcl'],
                        default='pcl_test', help='type of the observed point clouds')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = 'dataset'
    seq_len = 17
    loss_type = 'chamfer' if args.pcl_type == 'pcl_test' else 'p2s'

    seqs = [
        args.seq_name
    ]

    for p in seqs:
        modelname = '_'.join(p.split('_')[:-1])
        start_idx = int(p.split('_')[-1])
        gt_path = join(data_path, f'{modelname}_{start_idx}', args.pcl_type)
        out_dir = join(data_path, f'{modelname}_{start_idx}', 'h4d_fitting')
        back_optim(mesh_dir=out_dir,
                   device=device,
                   num_iterations=500)

