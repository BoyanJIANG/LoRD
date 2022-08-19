import torch
import torch.nn as nn
import torch.optim as optim
import os
from os.path import join
import numpy as np
import glob
import time
import trimesh
from tensorboardX import SummaryWriter
from smplx import SMPL

from lib.utils import gradient, vec_normalize as normalize
from lib.utils.point_utils import read_point_ply
from lib.utils import SmoothedValue
from lib.utils.h4d_utils import Prior, get_loss_weights, backward_step, compute_p2s_loss

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance


class Reconstructor():
    def __init__(
            self,
            model,
            tex_model,
            model_name,
            start_idx,
            data_folder,
            ply_folder,
            version,
            part_size,
            length_sequence=17,
            device=None,
            save_path=None,
            out_dir=None,
            use_h4d_smpl=True
    ):
        self.model = model
        self.tex_model = tex_model
        self.w_texture = True if self.tex_model is not None else False
        self.use_h4d_smpl = use_h4d_smpl
        self.device = device
        self.out_dir = out_dir
        self.save_path = save_path
        self.logger = SummaryWriter(save_path.replace('latent_vecs', 'logs'))

        self.num_optim_samples = 10000
        self.lr = 1e-3
        self.part_radius = part_size
        self.padding = 0.0
        self.latent_size = 128
        self.init_std = 0.01
        self.points_sigma = 0.01
        self.num_overlaps = 4

        self.length_sequence = length_sequence
        self.data_folder = data_folder
        self.ply_folder = ply_folder
        self.model_name = model_name
        self.start_idx = start_idx

        self.version = version

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        bm_path = 'data/SMPL_NEUTRAL.pkl'
        self.bm = SMPL(model_path=bm_path).to(self.device)
        self.faces = np.load('data/smpl_faces.npy')
        self.faces_tensor = torch.from_numpy(self.faces.astype(np.int64)).to(device)

        # read the initial estimation
        h4d_lat = torch.load(join(self.data_folder, f'{model_name}_{start_idx}',
                                  'h4d_fitting', 'latent_vecs.pt'))
        self.pose = h4d_lat['pose_code'].to(device)
        self.beta = h4d_lat['id_code'].to(device)
        self.motion = h4d_lat['motion_code'].to(device)
        self.init_pose = self.pose.clone().detach()
        self.init_beta = self.beta.clone().detach()
        self.init_motion = self.motion.clone().detach()
        self.init_codes = torch.cat([self.init_pose, self.init_beta, self.init_motion], -1)
        self.pose.requires_grad = True
        self.beta.requires_grad = True
        self.motion.requires_grad = True

        # load latent codes and model parameters
        load_dict = torch.load(join(save_path.replace('_refine', ''), f'{model_name}_{start_idx}.pt'))
        self.model.load_state_dict(load_dict['model'])
        self.c_m = load_dict['c_m'].to(device)
        self.c_s = load_dict['c_s'].to(device)
        self.c_m.requires_grad = True
        self.c_s.requires_grad = True
        if self.w_texture:
            self.c_t = load_dict['c_t'].to(self.device)
            self.c_t.requires_grad = True
            self.tex_model.load_state_dict(load_dict['tex_model'])

        self.rot_mtxs, self.transls, self.bboxs = self.get_transf()
        self.rot_mtxs = torch.from_numpy(self.rot_mtxs).float().to(device)
        self.transls = torch.from_numpy(self.transls).float().to(device)

        self.n_parts = self.transls.shape[1]


    def load_input_ply(self):
        files = sorted(glob.glob(join(self.ply_folder, '*.ply')))
        vs = []
        vns = []
        vcs = []
        for f in files:
            pcl_data = read_point_ply(f, self.w_texture)
            v = pcl_data[0].astype(np.float32)
            n = pcl_data[1].astype(np.float32)
            c = pcl_data[2].astype(np.float32) / 255.0
            vs.append(v)
            vns.append(n)
            vcs.append(c)
        return vs, vns, vcs


    def random_point_sample_seq(self, n_steps, time_step=None):
        if time_step is None:
            time_step = np.random.choice(n_steps)
        n_pts = self.vs[time_step].shape[0]

        idx = np.random.randint(n_pts, size=min(n_pts, self.num_optim_samples))
        point_samples = self.vs[time_step][None, idx, :]
        point_n_samples = self.vns[time_step][None, idx, :]
        point_c_samples = self.vcs[time_step][None, idx, :]

        return point_samples, point_n_samples, point_c_samples, time_step


    def get_transf(self, use_refined=False):
        model_name = self.model_name
        start_idx = self.start_idx

        part_file = 'data/test_2127_parts.pt'
        meta = torch.load(part_file)
        face_idx = meta['face_idx']
        alpha = meta['alpha']

        rot_mtxs = []
        transls = []
        bboxs = []

        if not use_refined:
            smpl_path = join(self.data_folder, f'{model_name}_{start_idx}', 'h4d_fitting')
        else:
            smpl_path = join(self.data_folder, f'{model_name}_{start_idx}', 'h4d_fitting_refine')

        for i in range(self.length_sequence):
            mesh = trimesh.load(join(smpl_path, f'{i:04d}.ply'), process=False)
            bboxs.append(mesh.bounds)
            v = mesh.vertices[mesh.faces[face_idx]]
            points = (alpha[:, :, None] * v).sum(axis=1)
            transls.append(points)

            xx = normalize(v[:, 0] - points)
            yy = normalize(mesh.face_normals[face_idx])
            zz = normalize(np.cross(xx, yy))
            rot_mtx = np.stack([xx, yy, zz], axis=-1)  # 1, 3, 3
            rot_mtxs.append(rot_mtx)

        return np.array(rot_mtxs), np.array(transls), np.array(bboxs)



    def get_points(self, pc_input, bbox, local_sigma=0.01):
        batch_size, sample_size, dim = pc_input.shape

        sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)

        bbox = torch.from_numpy(bbox).float().to(self.device)
        bbox[0] -= 0.1
        bbox[1] += 0.1
        boxsize = bbox[1] - bbox[0]
        sample_global = torch.rand([batch_size, sample_size // 8, dim], device=pc_input.device)
        sample_global = boxsize * sample_global + bbox[0]

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample


    def interp_layer(self, pts, pts_normal=None, time_step=0):
        bs, n_pts = pts.shape[:2]

        # Calculate the distance from each parts for each point
        parts_center = self.transls[time_step].unsqueeze(0).repeat(n_pts, 1, 1)
        pts_repeated = pts[0].unsqueeze(1).repeat(1, self.n_parts, 1)
        d_matrix = torch.norm(pts_repeated - parts_center, dim=-1)  # n_pts, n_parts
        valid_part_idx = (d_matrix <= (self.part_radius + self.padding)).float()

        # For each point, if the number of parts that contain it is less than K = self.num_overlaps,
        # we choose the top-K nearest parts, else we randomly choose K parts.
        shuffle_idx = np.random.permutation(self.n_parts)
        new_idx = valid_part_idx[:, shuffle_idx].argsort(descending=True)[:, :self.num_overlaps].cpu().numpy()
        part_idx = shuffle_idx[new_idx]
        if valid_part_idx.sum(-1).min() < self.num_overlaps:
            iso_pts = (valid_part_idx.sum(-1) < self.num_overlaps).cpu().numpy()
            part_idx[iso_pts] = d_matrix[iso_pts].topk(self.num_overlaps, dim=-1, largest=False).indices.cpu().numpy()
        part_idx = part_idx[None]

        # We use mean weights to fuse the predictions from overlapped parts
        weights = torch.ones([bs, n_pts, self.num_overlaps]).to(self.device) / self.num_overlaps

        lat_m = self.c_m[[part_idx]]  # 1, n_pts, 3, latent_size
        lat_s = self.c_s[[part_idx]]  # 1, n_pts, 3, latent_size
        if self.w_texture:
            lat_t = self.c_t[[part_idx]]  # 1, n_pts, 3, latent_size
        else:
            lat_t = None

        rot_mtxs = self.rot_mtxs[time_step, part_idx]  # 1, n_pts, num_overlaps], 3, 3
        transls = self.transls[time_step, part_idx]  # 1, n_pts, num_overlaps, 3

        # Transform each point to local coordinate frame
        vs_homo = torch.cat([pts, torch.ones([bs, n_pts, 1]).to(self.device)], dim=-1)
        R_transpose = rot_mtxs.permute(0, 1, 2, 4, 3)
        R_transpose_T = torch.einsum('blijk, blikm -> blijm', R_transpose, transls.unsqueeze(-1))
        home_transf_mtxs = torch.cat([R_transpose, -R_transpose_T], dim=-1)
        xloc = torch.einsum('blijk, blk -> blij', home_transf_mtxs, vs_homo)  # 1, n_pts, 3, 3
        xloc = (xloc / self.part_radius).to(self.device)  # 1, n_pts, 3, 3

        if pts_normal is not None:
            vnloc = torch.einsum('blijk, blk -> blij', R_transpose, pts_normal)
            return lat_m, lat_s, lat_t, weights, xloc, vnloc
        else:
            return lat_m, lat_s, lat_t, weights, xloc


    def optimize_latent_code(self, num_steps=3000):
        print('Optimizing LoRD...')
        if self.w_texture:
            optimizer = optim.Adam([
                {'params': self.tex_model.parameters(), 'lr': self.lr},
                {'params': self.c_m, 'lr': self.lr},
                {'params': self.c_s, 'lr': self.lr},
                {'params': self.c_t, 'lr': self.lr}
            ])
            self.tex_model.train()
        else:
            optimizer = optim.Adam([self.c_m, self.c_s], lr=self.lr)

        self.model.eval()

        losses = {
            'loss': SmoothedValue(),
            'mnfld_loss': SmoothedValue(),
            'grad_loss': SmoothedValue(),
            'normals_loss': SmoothedValue(),
            'lat_loss': SmoothedValue()
        }

        if self.w_texture:
            losses['tex_loss'] = SmoothedValue()

        start = time.time()
        for s in range(num_steps):
            _, loss_dict = self.optimize_step(optimizer, n_steps=self.length_sequence)
            for k, value in loss_dict.items():
                loss_dict[k] = value.item()
                losses[k].update(value.item())
            if s % 10 == 0:
                print_string = f'Step-{s:06d} | {self.model_name}_{self.start_idx} | {self.n_parts}parts '
                for k, v in loss_dict.items():
                    print_string += f'| {k}={v:.4f} '
                print_string += f'| time={(time.time() - start):.3f}'
                print(print_string)

                for k, v in loss_dict.items():
                    self.logger.add_scalar('optimization/%s' % k, v, (s + 1))

            if s % 500 == 0:
                print('Saving checkpoint...')
                save_dict = {'model': self.model.state_dict(),
                             'c_m': self.c_m.detach().cpu(),
                             'c_s': self.c_s.detach().cpu(),
                             'rot_mtxs': self.rot_mtxs.detach().cpu(),
                             'transls': self.transls.detach().cpu()}
                if self.tex_model is not None:
                    save_dict['tex_model'] = self.tex_model.state_dict()
                    save_dict['c_t'] = self.c_t.detach().cpu()
                torch.save(save_dict, join(self.save_path, self.version))

        print('Saving checkpoint...')
        save_dict = {'model': self.model.state_dict(),
                     'c_m': self.c_m.detach().cpu(),
                     'c_s': self.c_s.detach().cpu(),
                     'rot_mtxs': self.rot_mtxs.detach().cpu(),
                     'transls': self.transls.detach().cpu()}
        if self.tex_model is not None:
            save_dict['tex_model'] = self.tex_model.state_dict()
            save_dict['c_t'] = self.c_t.detach().cpu()
        torch.save(save_dict, join(self.save_path, self.version))


    def optimize_step(self, optimizer, n_steps=17):
        loss_t0, loss_dict_t0 = self.get_loss(n_steps, time_step=0)
        loss_t, loss_dict_t = self.get_loss(n_steps)

        all_norm = torch.norm(torch.cat([self.c_m, self.c_s], -1), dim=-1).reshape(-1)
        loss_lat = all_norm[torch.abs(all_norm) > 1e-7].mean()
        loss = loss_t0 + loss_t + loss_lat * 1e-3

        loss_dict = {}
        for k, v in loss_dict_t0.items():
            loss_dict[k] = loss_dict_t0[k] + loss_dict_t[k]
        loss_dict['lat_loss'] = loss_lat

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, loss_dict


    def get_loss(self, n_steps, time_step=None):
        mnfld_pnts, normals, cols, time_step = self.random_point_sample_seq(n_steps=n_steps, time_step=time_step)
        nonmnfld_pnts = self.get_points(mnfld_pnts, self.bboxs[time_step], time_step)
        mnfld_pnts.requires_grad_()
        nonmnfld_pnts.requires_grad_()

        # Normalize time value to [0, 1]
        time_val = torch.from_numpy(np.array(
            time_step / (self.length_sequence - 1), dtype=np.float32)).repeat(1).to(self.device)

        # predict SDFs for on-surface points
        mnfld_lat_m, mnfld_lat_s, mnfld_lat_t, mnfld_weights, \
        mnfld_xloc, normal_loc = self.interp_layer(mnfld_pnts,
                                                   normals,
                                                   time_step=time_step)
        mnfld_pred = self.model(mnfld_xloc, time_val, mnfld_lat_m, mnfld_lat_s)
        mnfld_pred_interp = (mnfld_pred * mnfld_weights.unsqueeze(-1)).sum(dim=2, keepdim=True)

        # predict RGB values
        if self.w_texture:
            bs, n_pts, n_neighbors, _ = mnfld_xloc.shape
            color_feat = torch.cat([mnfld_xloc, time_val.reshape(bs, 1, 1, 1).repeat(1, n_pts, n_neighbors, 1)], dim=-1)
            pred_cols = self.tex_model(color_feat, mnfld_lat_t)
            pred_cols_interp = (pred_cols * mnfld_weights.unsqueeze(-1)).sum(dim=2, keepdim=True)
            pred_cols = torch.cat([pred_cols, pred_cols_interp], dim=2)  # 1, npts, 5, 3

        # predict SDFs for off-surface points
        nonmnfld_lat_m, nonmnfld_lat_s, _, nonmnfld_weights, nonmnfld_xloc = self.interp_layer(nonmnfld_pnts,
                                                                                            time_step=time_step)
        nonmnfld_pred = self.model(nonmnfld_xloc, time_val, nonmnfld_lat_m, nonmnfld_lat_s)

        mnfld_grad = gradient(mnfld_xloc, mnfld_pred)
        nonmnfld_grad = gradient(nonmnfld_xloc, nonmnfld_pred)

        # Losses
        loss_dict = dict()
        # surface loss
        mnfld_loss = (torch.cat([mnfld_pred, mnfld_pred_interp], dim=2).abs()).mean()

        # eikonal loss
        grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

        loss = 1.0 * mnfld_loss + 0.1 * grad_loss

        # normal loss
        normals_loss = ((mnfld_grad - normal_loc).abs()).norm(2, dim=-1).mean()
        loss = loss + 1.0 * normals_loss

        # texture loss
        if self.w_texture:
            gt_cols = cols.unsqueeze(2).expand(pred_cols.shape)
            tex_loss = nn.L1Loss()(pred_cols, gt_cols)
            loss = loss + 1.0 * tex_loss
            loss_dict['tex_loss'] = tex_loss

        loss_dict.update({
            'loss': loss,
            'mnfld_loss': mnfld_loss,
            'grad_loss': grad_loss,
            'normals_loss': normals_loss,
        })

        return loss, loss_dict


    def optimize_h4d(self, num_steps):
        print('Optimizing H4D...')
        optimizer = optim.Adam([self.pose, self.beta, self.motion], lr=5e-3)

        self.model.eval()
        start = time.time()
        for s in range(num_steps):
            loss_dict = self.optimize_step_h4d(optimizer)
            for k, value in loss_dict.items():
                loss_dict[k] = value.item()
            if s % 10 == 0:
                print_string = f'Step-{s:06d} | {self.model_name}_{self.start_idx} | {self.n_parts}parts '
                for k, v in loss_dict.items():
                    print_string += f'| {k}={v:.6f} '
                print_string += f'| time={(time.time() - start):.3f}'
                print(print_string)

            if s % 500 == 0:
                print('Saving checkpoint...')
                save_dict = {'model': self.model.state_dict(),
                             'c_m': self.c_m.detach().cpu(),
                             'c_s': self.c_s.detach().cpu(),
                             'rot_mtxs': self.rot_mtxs.detach().cpu(),
                             'transls': self.transls.detach().cpu()}
                if self.tex_model is not None:
                    save_dict['tex_model'] = self.tex_model.state_dict()
                    save_dict['c_t'] = self.c_t.detach().cpu()
                torch.save(save_dict, join(self.save_path, self.version))

                self.visualize_smpl()

        print('Saving checkpoint...')
        save_dict = {'model': self.model.state_dict(),
                     'c_m': self.c_m.detach().cpu(),
                     'c_s': self.c_s.detach().cpu(),
                     'rot_mtxs': self.rot_mtxs.detach().cpu(),
                     'transls': self.transls.detach().cpu()}
        if self.tex_model is not None:
            save_dict['tex_model'] = self.tex_model.state_dict()
            save_dict['c_t'] = self.c_t.detach().cpu()
        torch.save(save_dict, join(self.save_path, self.version))


    def pca_layer(self, n_steps, time_step=None):
        pca = np.load('data/pca.npz')
        body_pca = torch.Tensor(pca['body_comp'][:256]).cuda()
        body_mean = torch.Tensor(pca['body_mean']).cuda()
        global_rot_pca = torch.Tensor(pca['global_comp'][:16]).cuda()
        global_rot_mean = torch.Tensor(pca['global_mean']).cuda()

        c_i_batch = self.beta.repeat(n_steps, 1)

        delta_root_orient = torch.matmul(self.motion[:, :16], global_rot_pca) + global_rot_mean
        delta_root_orient = delta_root_orient.view(n_steps-1, -1)
        delta_body_pose = torch.matmul(self.motion[:, 16:], body_pca) + body_mean
        delta_body_pose = delta_body_pose.view(n_steps-1, -1)

        delta_pose = torch.cat([delta_root_orient, delta_body_pose], -1)
        poses = self.pose + delta_pose  # 16, 72
        poses_stage1 = torch.cat([self.pose, poses], 0)  # 17, 72

        pred_verts = self.bm(
            betas=c_i_batch,
            body_pose=poses_stage1[:, 3:],
            global_orient=poses_stage1[:, :3],
        ).vertices

        if time_step is None:
            time_step = np.random.choice(n_steps)

        return pred_verts, poses_stage1, time_step


    def optimize_step_h4d(self, optimizer, time_step=None):
        smpl_verts, _, time_step = self.pca_layer(n_steps=self.length_sequence, time_step=time_step)  # 1, 6890, 3
        smpl_verts = smpl_verts[[time_step]]
        # --------------------------------------------------------------------------------------
        lat_m, lat_s, _, weights, xloc = self.interp_layer(smpl_verts, time_step=time_step)

        time_val = torch.from_numpy(np.array(
            time_step / (17 - 1), dtype=np.float32)).repeat(1).to(self.device)

        pred = self.model(xloc, time_val, lat_m, lat_s)

        pred_interp = (pred * weights.unsqueeze(-1)).sum(dim=2, keepdim=True)
        pred = torch.cat([pred, pred_interp], dim=2)  # 1*npoints*9
        q_x = pred.abs()
        w1 = torch.ones_like(pred).float()
        w2 = torch.ones_like(pred).float() * 0.2
        loss_weight = torch.where(pred >= 0, w1, w2)
        loss_smpl = (q_x * loss_weight).mean()
        # --------------------------------------------------------------------------------------
        pred_mesh = Meshes(verts=[smpl_verts[0]],
                           faces=[self.faces_tensor])
        gt_pcl = Pointclouds(points=[self.vs[time_step]], normals=[self.vns[time_step]])
        loss_p2s = compute_p2s_loss(pred_mesh, gt_pcl, seq_len=1)
        # --------------------------------------------------------------------------------------
        loss_reg = torch.norm(torch.cat([self.pose, self.beta, self.motion], -1) - self.init_codes, dim=-1).mean()
        # --------------------------------------------------------------------------------------

        loss = loss_smpl + 1e2 * loss_p2s + 1e-3 * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict = {
            'loss': loss,
            'loss_smpl': loss_smpl,
            'loss_p2s': loss_p2s,
            'loss_reg': loss_reg,
        }

        return loss_dict



    def joint_optimization(self):

        # ------ Optimization ------
        self.optimize_h4d(num_steps=3000)
        self.visualize_smpl()

        self.rot_mtxs, self.transls, self.bboxs = self.get_transf(use_refined=True)
        self.rot_mtxs = torch.from_numpy(self.rot_mtxs).float().to(self.device)
        self.transls = torch.from_numpy(self.transls).float().to(self.device)
        self.optimize_latent_code(num_steps=3000)


    def visualize_smpl(self):
        verts, pred_pose, _ = self.pca_layer(self.length_sequence)
        vis_dir = join(self.data_folder, f'{self.model_name}_{self.start_idx}', 'h4d_fitting_refine')
        os.makedirs(vis_dir, exist_ok=True)
        for i, v in enumerate(verts.detach().cpu().numpy()):
            mesh = trimesh.Trimesh(v, self.faces, process=False)
            mesh.export(join(vis_dir, f'{i:04d}.ply'))

        # -----------save codes-------------
        print('Saving latent vectors...')
        torch.save(
            {"id_code": self.beta.detach().cpu(),
             "pose_code": self.pose.detach().cpu(),
             "motion_code": self.motion.detach().cpu()},
            os.path.join(vis_dir, 'latent_vecs.pt')
        )

        np.savez(os.path.join(vis_dir, 'smpl_params.npz'),
                 beta=self.beta.detach().cpu().numpy(),
                 poses=pred_pose.detach().cpu().numpy())


    def run(self):
        self.vs, self.vns, self.vcs = self.load_input_ply()
        for i, (v, n, c) in enumerate(zip(self.vs, self.vns, self.vcs)):
            shuffle_index = np.random.permutation(v.shape[0])
            self.vs[i] = torch.from_numpy(v[shuffle_index]).to(self.device)
            self.vns[i] = torch.from_numpy(n[shuffle_index]).to(self.device)
            self.vcs[i] = torch.from_numpy(c[shuffle_index]).to(self.device)

        print('Optimizing latent codes...')
        self.joint_optimization()
