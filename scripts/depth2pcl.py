import os
from tqdm import tqdm
from os.path import join
import torch
from torch.nn import functional as F
import numpy as np
import cv2
from lib.utils.point_utils import write_point_ply


def compute_normal_from_depth(depth, mask, K_inv, xyz, apply_mask=True):
    # depth,mask:[B,1,H,W]
    mask = mask.to(torch.bool)
    batch, _, height, width = depth.shape
    # [B,3,3]x[B,3,HW]->[B,3,HW]->[B,3,H,W]
    cam_r = (K_inv @ xyz).reshape(batch, 3, height, width)
    cam_p = cam_r * depth  # [B,3,H,W]

    pxd = F.pad(depth, [1, 1, 0, 0])
    pyd = F.pad(depth, [0, 0, 1, 1])
    px = F.pad(cam_p, [1, 1, 0, 0])
    py = F.pad(cam_p, [0, 0, 1, 1])
    pxm = F.pad(mask, [1, 1, 0, 0])
    pym = F.pad(mask, [0, 0, 1, 1])

    # compute valid mask for normal
    mx1 = (pxm[:, :, :, 2:] & pxm[:, :, :, 1:-1])
    mx2 = (pxm[:, :, :, :-2] & pxm[:, :, :, 1:-1])
    my1 = (pym[:, :, 2:] & pym[:, :, 1:-1])
    my2 = (pym[:, :, :-2] & pym[:, :, 1:-1])

    # filter mask by limited depth diff
    ddx1 = (pxd[:, :, :, 2:] - pxd[:, :, :, 1:-1])
    ddx2 = (pxd[:, :, :, 1:-1] - pxd[:, :, :, :-2])
    ddy1 = (pyd[:, :, 2:] - pyd[:, :, 1:-1])
    ddy2 = (pyd[:, :, 1:-1] - pyd[:, :, :-2])
    if (apply_mask):
        mx1 = mx1 & (ddx1.abs() < pxd[:, :, :, 1:-1] * 0.05)
        mx2 = mx2 & (ddx2.abs() < pxd[:, :, :, 1:-1] * 0.05)
        my1 = my1 & (ddy1.abs() < pyd[:, :, 1:-1] * 0.05)
        my2 = my2 & (ddy2.abs() < pyd[:, :, 1:-1] * 0.05)
    mx = mx1 | mx2
    my = my1 | my2
    m = mx & my

    # compute finite diff gradients
    dx1 = (px[:, :, :, 2:] - px[:, :, :, 1:-1])
    dx2 = (px[:, :, :, 1:-1] - px[:, :, :, :-2])
    dy1 = (py[:, :, 2:] - py[:, :, 1:-1])
    dy2 = (py[:, :, 1:-1] - py[:, :, :-2])
    # dxs = dx1 * mx1 + dx2 * (mx2 & ~mx1)
    # dys = dy1 * my1 + dy2 * (my2 & ~my1)
    dxs = (dx1 * mx1 + dx2 * mx2) / (mx1.float() + mx2.float() + (~mx).float())
    dys = (dy1 * my1 + dy2 * my2) / (my1.float() + my2.float() + (~my).float())
    dx = F.normalize(dxs, dim=1, p=2)
    dy = F.normalize(dys, dim=1, p=2)

    # compute normal direction from cross products
    normal = torch.cross(dy, dx, dim=1)

    # flip normal based on camera view
    normal = F.normalize(normal, p=2, dim=1)
    cam_dir = F.normalize(cam_r, p=1, dim=1)

    dot = (cam_dir * normal).sum(1, keepdim=True)
    normal *= -dot.sign()

    if (apply_mask):
        nm = m.repeat(1, 3, 1, 1)
        normal[~nm] = 0

    return cam_p, normal


if __name__ == '__main__':
    data_folder = 'data/demo'
    out_folder = 'dataset'
    seq_len = 17

    seqs = [
        '00134_longlong_twist_trial2_21',
        '02474_longshort_ROM_lower_258'
    ]

    for p in tqdm(seqs):
        p = {'model': '_'.join(p.split('_')[:-1]),
             'start_idx': int(p.split('_')[-1])}
        model_name = f"{p['model']}_{p['start_idx']}"
        for i in range(seq_len):
            img_res = 512
            fx = fy = 1 / (32/35/img_res)
            K = np.array([
                [fx, 0, img_res/2],
                [0, fy, img_res/2],
                [0, 0, 1]
            ]).astype(np.float32)
            K_inv = torch.from_numpy(np.linalg.inv(K)).float()

            # depth image, uint16, unit: 1/1000 meter
            raw_depth = cv2.imread(join(data_folder, model_name, 'rendering', f'{i:04d}.png'), -1) / 1000.
            raw_mask = raw_depth > 0
            valid = raw_mask.reshape(-1)

            depth = torch.from_numpy(raw_depth[None, None, :, :]).float()
            mask = torch.from_numpy(raw_mask[None, None, :, :]).float()
            u = range(0, img_res)
            v = range(0, img_res)
            x, y = np.meshgrid(u, v)
            z = np.ones_like(x)
            x = np.ravel(x)
            y = np.ravel(y)
            z = np.ravel(z)
            xyz = torch.from_numpy(np.stack([x, y, z], 0)[None]).float()
            pts, normal = compute_normal_from_depth(depth, mask, K_inv, xyz)
            pts = pts[0].permute(1, 2, 0).reshape(-1, 3)[valid].numpy()
            pts[:, 2] -= 3.5  # camera translation
            normal = normal[0].permute(1, 2, 0).reshape(-1, 3)[valid].numpy()

            # transform to the original SMPL space
            R = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]).astype(np.float32)
            pts = (R @ pts.T).T
            normal = (R @ normal.T).T

            save_folder = join(out_folder, model_name, 'depth_pcl')
            os.makedirs(save_folder, exist_ok=True)
            write_point_ply(join(save_folder, f'{i:04d}.ply'), pts, normal)
