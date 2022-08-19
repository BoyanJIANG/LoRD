import torch
import os
from os.path import join
import numpy as np
import time
import datetime

from lib.utils import gradient, SmoothedValue
from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(
            self,
            model,
            data_loader,
            optimizer,
            c_m,
            c_s,
            n_parts,
            part_size,
            device=None,
            save_path=None,
    ):
        self.model = model
        self.c_m = c_m
        self.c_s = c_s
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.save_path = save_path

        self.part_radius = part_size
        self.padding = 0.0
        self.num_overlaps = 4
        self.n_parts = n_parts

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.logger = SummaryWriter(os.path.join(save_path, 'logs'))


    def get_points(self, pc_input, bbox, local_sigma=0.01):
        batch_size, sample_size, dim = pc_input.shape

        sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
        bbox[0] -= 0.1
        bbox[1] += 0.1
        boxsize = bbox[1] - bbox[0]
        sample_global = torch.rand([batch_size, sample_size // 8, dim], device=pc_input.device)
        sample_global = boxsize * sample_global + bbox[0]

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample


    def interp_layer(self, c_m, c_s, pts, rot_mtx, transl, pts_normal=None):
        '''
        :param c_m: n_parts, latent_size
        :param c_s: n_parts, latent_size
        :param pts: 1, n_optim_points, 3
        :param rot_mtx: 1, n_parts, 3, 3
        :param transl: 1, n_parts, 3
        '''
        bs, n_pts = pts.shape[:2]

        # Calculate the distance from each parts for each point
        parts_center = transl.unsqueeze(0).repeat(n_pts, 1, 1)
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

        lat_m = c_m[[part_idx]]
        lat_s = c_s[[part_idx]]

        # Transform each point to local coordinate frame
        rot_mtxs = rot_mtx[[part_idx]]  # 1, n_pts, 3, 3, 3
        transls = transl[[part_idx]]  # 1, n_pts, 3, 3
        vs_homo = torch.cat([pts, torch.ones([bs, n_pts, 1]).to(self.device)], dim=-1)
        R_transpose = rot_mtxs.permute(0, 1, 2, 4, 3)
        R_transpose_T = torch.einsum('blijk, blikm -> blijm', R_transpose, transls.unsqueeze(-1))
        home_transf_mtxs = torch.cat([R_transpose, -R_transpose_T], dim=-1)
        xloc = torch.einsum('blijk, blk -> blij', home_transf_mtxs, vs_homo)  # 1, n_pts, 3, 3
        # normalize to [-1, 1]
        xloc = (xloc / self.part_radius).to(self.device)  # 1, n_pts, 3, 3

        if pts_normal is not None:
            vnloc = torch.einsum('blijk, blk -> blij', R_transpose, pts_normal)
            return lat_m, lat_s, weights, xloc, vnloc
        else:
            return lat_m, lat_s, weights, xloc


    def get_loss(self, data, is_t0):
        if is_t0:
            mnfld_pnts = data['v_0'].to(self.device)
            normals = data['vn_0'].to(self.device)
            bbox = data['bbox_0'][0].to(self.device)
            rot_mtx = data['rot_mtx_0'][0].to(self.device)
            transl = data['transl_0'][0].to(self.device)
            time_val = data['time_val_0'].to(self.device)
        else:
            mnfld_pnts = data['v_t'].to(self.device)
            normals = data['vn_t'].to(self.device)
            bbox = data['bbox_t'][0].to(self.device)
            rot_mtx = data['rot_mtx_t'][0].to(self.device)
            transl = data['transl_t'][0].to(self.device)
            time_val = data['time_val_t'].to(self.device)
        idx = data['idx'][0]
        c_m = self.c_m[idx]
        c_s = self.c_s[idx]

        nonmnfld_pnts = self.get_points(mnfld_pnts, bbox)
        mnfld_pnts.requires_grad_()
        nonmnfld_pnts.requires_grad_()

        mnfld_lat_m, mnfld_lat_s, mnfld_weights, \
        mnfld_xloc, normal_loc = self.interp_layer(c_m, c_s, mnfld_pnts,
                                                   rot_mtx, transl, normals)
        mnfld_pred = self.model(mnfld_xloc, time_val, mnfld_lat_m, mnfld_lat_s)
        mnfld_pred_interp = (mnfld_pred * mnfld_weights.unsqueeze(-1)).sum(dim=2, keepdim=True)

        nonmnfld_lat_m, nonmnfld_lat_s, nonmnfld_weights,\
        nonmnfld_xloc = self.interp_layer(c_m, c_s, nonmnfld_pnts,
                                           rot_mtx, transl)
        nonmnfld_pred = self.model(nonmnfld_xloc, time_val, nonmnfld_lat_m, nonmnfld_lat_s)

        mnfld_grad = gradient(mnfld_xloc, mnfld_pred)
        nonmnfld_grad = gradient(nonmnfld_xloc, nonmnfld_pred)

        # surface loss
        mnfld_loss = (torch.cat([mnfld_pred, mnfld_pred_interp], dim=2).abs()).mean()

        # eikonal loss
        grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
        loss = 1.0 * mnfld_loss + 0.1 * grad_loss

        # normal loss
        normals_loss = ((mnfld_grad - normal_loc).abs()).norm(2, dim=-1).mean()
        loss = loss + 1.0 * normals_loss

        # latent regularization
        all_norm = torch.norm(torch.cat([c_m, c_s], -1), dim=-1).reshape(-1)
        lat_loss = all_norm[torch.abs(all_norm) > 1e-7].mean()
        loss = loss + 1e-3 * lat_loss

        loss_dict = {
            'loss': loss,
            'mnfld_loss': mnfld_loss,
            'grad_loss': grad_loss,
            'normals_loss': normals_loss,
            'lat_loss': lat_loss,
        }

        return loss, loss_dict


    def run(self):
        epoch = 0
        print_every = 10
        save_every = 500
        while True:
            self.model.train()

            losses = {
                'loss': SmoothedValue(),
                'mnfld_loss': SmoothedValue(),
                'grad_loss': SmoothedValue(),
                'normals_loss': SmoothedValue(),
                'lat_loss': SmoothedValue()
            }

            batch_time = SmoothedValue()

            start = time.time()
            end = time.time()

            for iter, data in enumerate(self.data_loader):
                self.optimizer.zero_grad()

                loss_t0, loss_dict_t0 = self.get_loss(data, is_t0=True)
                loss_t, loss_dict_t = self.get_loss(data, is_t0=False)

                total_loss = loss_t0 + loss_t

                t_batch = time.time() - end
                end = time.time()
                batch_time.update(t_batch)
                eta_seconds = batch_time.global_avg * (len(self.data_loader) - iter + 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                total_time_string = str(datetime.timedelta(seconds=int(time.time() - start)))

                total_loss.backward()
                self.optimizer.step()

                loss_dict = {}
                for k, v in loss_dict_t0.items():
                    loss_dict[k] = loss_dict_t0[k] + loss_dict_t[k]

                for k, v in loss_dict.items():
                    losses[k].update(v.item())

                if (iter + 1) % print_every == 0:
                    summary_string = f'Epoch {epoch + 1} ({iter + 1}/{len(self.data_loader)}) | ' \
                                     f'{self.n_parts} parts | ' \
                                     f'Total: {total_time_string} | ' \
                                     f'ETA: {eta_string} | '

                    for k, v in loss_dict.items():
                        summary_string += f' | {k}: {losses[k].avg:.4f}'
                        self.logger.add_scalar('train/%s' % k, v, (iter + 1))

                    print(summary_string)

                if (iter + 1) % save_every == 0:
                    print('Saving checkpoint...')
                    torch.save({'c_m': self.c_m.detach().cpu(),
                                'c_s': self.c_s.detach().cpu(),
                                'model': self.model.state_dict()},
                               join(self.save_path, 'model.pt'))


            print('Saving checkpoint...')
            torch.save({'c_m': self.c_m.detach().cpu(),
                        'c_s': self.c_s.detach().cpu(),
                        'model': self.model.state_dict()},
                       join(self.save_path, f'model_{epoch+1}.pt'))
            epoch += 1


