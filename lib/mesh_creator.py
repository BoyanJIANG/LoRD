import numpy as np
import plyfile
import time
import torch
import os
from tqdm import tqdm
import trimesh
import mcubes

from lib.utils.postprocess_utils import remove_backface
from lib.utils.point_utils import read_point_ply



def interp_layer(latent_vecs, pts, time_step, rot_mtxs, transls, radius):
    least_neighbour = 4
    device = pts.device
    bs, n_pts = pts.shape[:2]
    n_parts = rot_mtxs.shape[1]

    parts_center = transls[time_step].unsqueeze(0).repeat(n_pts, 1, 1)
    pts_repeated = pts[0].unsqueeze(1).repeat(1, n_parts, 1)
    d_matrix = torch.norm(pts_repeated - parts_center, dim=-1)
    valid_parts = d_matrix <= radius

    # record the max number of parts containing the same point for batch processing
    max_n_valid_parts = valid_parts.sum(-1).max()
    max_n_valid_parts = max_n_valid_parts.clamp_max(32)
    n_neighbour = max(least_neighbour, max_n_valid_parts)

    knn = d_matrix.topk(n_neighbour, dim=-1, largest=False)
    dist = knn.values[None, ...]
    part_idx = knn.indices[None, ...]

    # mean weight
    # different from training or optimization
    # we need to capture the geometry that far from SMPL body, e.g. blazer
    valid_weight = dist <= radius
    valid_weight[valid_weight.sum(-1) == 0] = 1
    weights = valid_weight / (torch.sum(valid_weight, dim=-1, keepdim=True) + 1e-8)

    lat = latent_vecs[part_idx]

    R = rot_mtxs[time_step, part_idx]
    T = transls[time_step, part_idx]

    vs_homo = torch.cat([pts, torch.ones([bs, n_pts, 1]).to(device)], dim=-1)
    R_transpose = R.permute(0, 1, 2, 4, 3)
    R_transpose_T = torch.einsum('blijk, blikm -> blijm', R_transpose, T.unsqueeze(-1))
    home_transf_mtxs = torch.cat([R_transpose, -R_transpose_T], dim=-1)
    xloc = torch.einsum('blijk, blk -> blij', home_transf_mtxs, vs_homo)
    xloc = (xloc / radius).to(device)

    return lat, weights, xloc



def create_mesh(
        data,
        model,
        tex_model,
        latent_vecs,
        tex_vec,
        time_step,
        filename,
        bbox,
        crop_size,
        pcl_path,
        max_batch=20000,
        radius=0.05
):
    """
    This function is adapted from DeepSDF: https://github.com/facebookresearch/DeepSDF
    """
    ply_filename = filename

    model.eval()

    time_val = torch.from_numpy(np.array(
        time_step / (17 - 1), dtype=np.float32)).repeat(1).cuda()

    # pad bounding box
    xmin = bbox[0].round(4) - 0.1
    xmax = bbox[1].round(4) + 0.1
    r = (xmax - xmin) / crop_size
    output_grid_shape = np.ceil(r).astype(np.int32)

    # create volume
    l = [np.linspace(xmin[i], xmax[i], output_grid_shape[i]) for i in range(3)]
    samples = torch.from_numpy(np.stack(np.meshgrid(l[0], l[1], l[2],
                                        indexing='ij'), axis=-1).astype(np.float32)).reshape(-1, 3)
    sdf_values = torch.ones(*output_grid_shape).flatten()
    samples = torch.cat([samples, sdf_values.unsqueeze(-1)], -1)
    samples.requires_grad = False

    head = 0
    iters = 0
    start = time.time()

    n_pts = samples.shape[0]
    total_iters = int(np.ceil(n_pts / max_batch))

    print('Processing points...')
    while head < n_pts:
        query_points = samples[None, head:min(head + max_batch, n_pts), :3].cuda()

        lat, weights, xloc = interp_layer(latent_vecs,
                                             query_points,
                                             time_step,
                                             data['rot_mtxs'].cuda(),
                                             data['transls'].cuda(),
                                             radius=radius)
        n_parts = xloc.shape[2]

        with torch.no_grad():
            pred = model(xloc, time_val, lat[..., :128],
                         lat[..., 128:]).squeeze(-1)

        samples[head:min(head + max_batch, n_pts), 3] = \
            (pred * weights).sum(dim=2).float().detach().cpu()

        head += max_batch
        iters += 1
        print(f'Iter {iters}/{total_iters}, {n_parts} parts')

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(*output_grid_shape)

    eval_end = time.time()
    print("evaluate sdf values takes: %f" % (eval_end - start))

    convert_sdf_samples_to_ply(
        data, radius,
        tex_model, tex_vec,
        sdf_values.data.cpu(),
        xmin,
        xmax,
        output_grid_shape,
        time_step,
        pcl_path,
        ply_filename
    )

    print("total takes: %f" % (time.time() - start))



def convert_sdf_samples_to_ply(
        data,
        radius,
        tex_model,
        tex_vec,
        pytorch_3d_sdf_tensor,
        xmin,
        xmax,
        grid_shape,
        time_step,
        pcl_path,
        ply_filename_out,
):
    """
    Convert sdf samples to trimesh, and predict RGB values for each mesh vertex.
    This function is adapted from DeepSDF: https://github.com/facebookresearch/DeepSDF
    """

    numpy_3d_sdf_tensor = -pytorch_3d_sdf_tensor.numpy()

    vertices, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, 0)
    mesh_points = vertices / (grid_shape - 1.0) * (xmax - xmin) + xmin

    mesh = trimesh.Trimesh(mesh_points, faces)

    # borrowed from LIG: https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/projects/local_implicit_grid/README.md
    print('Postprocessing generated mesh...')
    v, n, _ = read_point_ply(os.path.join(pcl_path, '%04d.ply' % time_step))
    surface_points = np.concatenate([v, n], axis=1)
    mesh = remove_backface(mesh, surface_points)
    mesh.export(ply_filename_out + '.ply')


    # query mesh vertices to the texture model
    if tex_vec is not None:
        mesh_v = torch.from_numpy(mesh.vertices).float().cuda()
        chunks = torch.split(torch.arange(mesh_v.shape[0]), 5000)
        all_pred_cols = []
        for chunk_idx in tqdm(chunks):
            lat, weights, xloc = interp_layer(tex_vec,
                                                mesh_v[None, chunk_idx],
                                                time_step,
                                                data['rot_mtxs'].cuda(),
                                                data['transls'].cuda(),
                                                radius=radius)
            bs, n_pts, n_neighbors, _ = xloc.shape
            time_val = torch.from_numpy(np.array(
                time_step / (17 - 1), dtype=np.float32)).repeat(1).cuda()
            color_feat = torch.cat([xloc, time_val.reshape(bs, 1, 1, 1).repeat(1, n_pts, n_neighbors, 1)], dim=-1)
            pred_cols = tex_model(color_feat, lat)[0]
            pred_cols_interp = (pred_cols * weights.unsqueeze(-1)).sum(dim=2, keepdim=True).squeeze()
            pred_cols_interp = pred_cols_interp.clamp(0, 1)
            all_pred_cols.append(pred_cols_interp)
        pred_cols_interp = torch.cat(all_pred_cols).detach().cpu().numpy()
        mesh_tex = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=pred_cols_interp, process=False)
        mesh_tex.export(ply_filename_out + '_tex.ply')


