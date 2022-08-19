import torch
import numpy as np
from plyfile import PlyData
from plyfile import PlyElement
from scipy.spatial import cKDTree
import trimesh
import torch.nn.functional as F



def sample_mesh(mesh, sampling_density):
  """Samples oriented points from a mesh."""
  num_samples = int(mesh.area / sampling_density)
  sample_pts, sample_face_ids = trimesh.sample.sample_surface(mesh, num_samples)
  sample_normals = mesh.face_normals[sample_face_ids]
  return sample_pts, sample_normals


def read_point_ply(filename, w_texture=False):
    """Load point cloud from ply file.

    Args:
      filename: str, filename for ply file to load.
    Returns:
      v: np.array of shape [#v, 3], vertex coordinates
      n: np.array of shape [#v, 3], vertex normals
      c: np.array of shape [#v, 3], vertex colors
    """
    pd = PlyData.read(filename)['vertex']
    v = np.array(np.stack([pd[i] for i in ['x', 'y', 'z']], axis=-1))
    n = np.array(np.stack([pd[i] for i in ['nx', 'ny', 'nz']], axis=-1))
    if w_texture:
        c = np.array(np.stack([pd[i] for i in ['red', 'green', 'blue']], axis=-1))
    else:
        c = np.zeros_like(v).astype(np.float32)
    return v, n, c


def write_point_ply(filename, v, n):
    """Write point cloud to ply file.

    Args:
      filename: str, filename for ply file to load.
      v: np.array of shape [#v, 3], vertex coordinates
      n: np.array of shape [#v, 3], vertex normals
    """
    vn = np.concatenate([v, n], axis=1)
    vn = [tuple(vn[i]) for i in range(vn.shape[0])]
    vn = np.array(vn, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    el = PlyElement.describe(vn, 'vertex')
    PlyData([el]).write(filename)


def sample_points_from_ray(points, normals, sample_factor=10, std=0.01):
    """Get sample points from points from ray.

    Args:
      points (numpy array): [npts, 3], xyz coordinate of points on the mesh surface.
      normals (numpy array): [npts, 3], normals of points on the mesh surface.
      sample_factor (int): number of samples to pick per surface point.
      std (float): std of samples to generate.
    Returns:
      points (numpy array): [npts*sample_factor, 3], where last dimension is
      distance to surface point.
      sdf_values (numpy array): [npts*sample_factor, 1], sdf values of the sampled points
      near the mesh surface.
    """
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    npoints = points.shape[0]
    offsets = np.random.randn(npoints, sample_factor, 1) * std
    point_samples = points[:, np.newaxis, :] + offsets * normals[:, np.newaxis, :]
    point_samples = point_samples.reshape(-1, points.shape[1])
    sdf_values = offsets.reshape(-1)
    point_samples = point_samples.astype(np.float32)
    sdf_values = sdf_values.astype(np.float32)
    return point_samples, sdf_values


def sample_points_from_ray_with_pad(points, normals, n_target, std=0.01):
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    npoints = points.shape[0]
    if npoints > n_target:
        print('We need bigger n_target!!!')
    sample_factor = int(np.ceil(float(n_target) / float(npoints)))
    offsets = np.random.randn(npoints, sample_factor, 1) * std
    point_samples = points[:, np.newaxis, :] + offsets * normals[:, np.newaxis, :]
    point_samples = point_samples.reshape(-1, points.shape[1])
    sdf_values = offsets.reshape(-1)
    point_samples = point_samples.astype(np.float32)
    sdf_values = sdf_values.astype(np.float32)

    # shuffle
    shuffle_index = np.random.permutation(point_samples.shape[0])[:n_target]
    point_samples = point_samples[shuffle_index]
    sdf_values = sdf_values[shuffle_index]

    return point_samples, sdf_values


def sample_points_from_ray_with_pad_torch(points, normals, n_target, std=0.01):
    normals = F.normalize(normals, dim=1)
    npoints = points.shape[0]
    if npoints > n_target:
        print('We need bigger n_target!!!')
    sample_factor = int(np.ceil(float(n_target) / float(npoints)))
    offsets = torch.randn(npoints, sample_factor, 1) * std
    point_samples = points.unsqueeze(1) + offsets * normals.unsqueeze(1)
    point_samples = point_samples.reshape(-1, points.shape[1])
    sdf_values = offsets.reshape(-1)
    point_samples = point_samples.float()
    sdf_values = sdf_values.float()

    # shuffle
    shuffle_index = np.random.permutation(point_samples.shape[0])[:n_target]
    point_samples = point_samples[shuffle_index]
    sdf_values = sdf_values[shuffle_index]

    return point_samples, sdf_values


def np_pad_points(points, ntarget):
    """Pad point cloud to required size.

    If number of points is larger than ntarget, take ntarget random samples.
    If number of points is smaller than ntarget, pad by repeating last point.
    Args:
      points: `[npoints, nchannel]` np array, where first 3 channels are xyz.
      ntarget: int, number of target channels.
    Returns:
      result: `[ntarget, nchannel]` np array, padded points to ntarget numbers.
    """
    if points.shape[0] < ntarget:
        mult = np.ceil(float(ntarget) / float(points.shape[0])) - 1
        rand_pool = np.tile(points, [int(mult), 1])
        nextra = ntarget - points.shape[0]
        extra_idx = np.random.choice(rand_pool.shape[0], nextra, replace=False)
        extra_pts = rand_pool[extra_idx]
        points_out = np.concatenate([points, extra_pts], axis=0)
    else:
        idx_choice = np.random.choice(points.shape[0], size=ntarget, replace=False)
        points_out = points[idx_choice]

    return points_out


def np_gather_ijk_index(arr, index):
    """Gather the features of given index from the feature grid.

    Args:
        arr (numpy array): h*w*d*c, feature grid.
        index (numpy array): nx*3, index of the feature grid
    Returns:
        nx*c, features at given index of the feature grid.
    """
    arr_flat = arr.reshape(-1, arr.shape[-1])
    _, j, k, _ = arr.shape
    index_transform = index[:, 0] * j * k + index[:, 1] * k + index[:, 2]
    return arr_flat[index_transform]


def np_shifted_crop(v, idx_grid, shift, crop_size, ntarget):
    """Crop the """
    nchannels = v.shape[1]
    vxyz = v[:, :3] - shift * crop_size * 0.5
    vall = v.copy()
    point_idxs = np.arange(v.shape[0])
    point_grid_idx = np.floor(vxyz / crop_size).astype(np.int32)
    valid_mask = np.ones(point_grid_idx.shape[0]).astype(np.bool)
    for i in range(3):
        valid_mask = np.logical_and(valid_mask, point_grid_idx[:, i] >= 0)
        valid_mask = np.logical_and(valid_mask, point_grid_idx[:, i] < idx_grid.shape[i])
    point_grid_idx = point_grid_idx[valid_mask]
    # translate to global grid index
    point_grid_idx = np_gather_ijk_index(idx_grid, point_grid_idx)

    vall = vall[valid_mask]
    point_idxs = point_idxs[valid_mask]
    crop_indices, revidx = np.unique(point_grid_idx, axis=0, return_inverse=True)
    ncrops = crop_indices.shape[0]
    sortarr = np.argsort(revidx)
    revidx_sorted = revidx[sortarr]
    vall_sorted = vall[sortarr]
    point_idxs_sorted = point_idxs[sortarr]
    bins = np.searchsorted(revidx_sorted, np.arange(ncrops))
    bins = list(bins) + [v.shape[0]]
    sid = bins[0:-1]
    eid = bins[1:]
    # initialize outputs
    point_crops = np.zeros([ncrops, ntarget, nchannels]).astype(np.float32)
    crop_point_idxs = []
    # extract crops and pad
    for i, (s, e) in enumerate(zip(sid, eid)):
        cropped_points = vall_sorted[s:e]
        crop_point_idx = point_idxs_sorted[s:e]
        crop_point_idxs.append(crop_point_idx)
        padded_points = np_pad_points(cropped_points, ntarget=ntarget)
        point_crops[i] = padded_points
    return point_crops, crop_indices, crop_point_idxs


def np_get_occupied_idx(v,
                        xmin=(0., 0., 0.),
                        xmax=(1., 1., 1.),
                        crop_size=.125,
                        ntarget=2048,
                        overlap=True,
                        normalize_crops=False,
                        return_shape=False,
                        return_crop_point_idxs=False):
    """Get crop indices for point clouds."""
    # v = v.copy() - xmin
    v = v.copy()
    v[:, :3] = v[:, :3] - xmin
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    r = (xmax - xmin) / crop_size
    r = np.ceil(r)
    rr = r.astype(np.int32) if not overlap else (2 * r - 1).astype(np.int32)
    # create index grid
    idx_grid = np.stack(np.meshgrid(np.arange(rr[0]), np.arange(rr[1]), np.arange(rr[2]), indexing='ij'), axis=-1)
    # [rr[0], rr[1], rr[2], 3]

    shift_idxs = np.stack(np.meshgrid(np.arange(int(overlap) + 1),
                                      np.arange(int(overlap) + 1),
                                      np.arange(int(overlap) + 1),
                                      indexing='ij'),
                          axis=-1)
    shift_idxs = np.reshape(shift_idxs, [-1, 3])
    point_crops = []
    crop_indices = []
    crop_point_idxs = []
    for i in range(shift_idxs.shape[0]):
        sft = shift_idxs[i]
        skp = int(overlap) + 1
        idg = idx_grid[sft[0]::skp, sft[1]::skp, sft[2]::skp]
        pc, ci, cpidx = np_shifted_crop(v, idg, sft, crop_size=crop_size, ntarget=ntarget)
        point_crops.append(pc)
        crop_indices.append(ci)
        crop_point_idxs += cpidx
    point_crops = np.concatenate(point_crops, axis=0)  # [ncrops, nsurface, 6]
    crop_indices = np.concatenate(crop_indices, axis=0)  # [ncrops, 3]

    if normalize_crops:
        # normalize each crop
        if overlap:
            crop_corners = crop_indices * 0.5 * crop_size
            crop_centers = crop_corners + 0.5 * crop_size  # [ncrops, 3]
        else:
            # add new branch here to fix bug..
            crop_corners = crop_indices * crop_size
            crop_centers = crop_corners + 0.5 * crop_size  # [ncrops, 3]

        crop_centers = crop_centers[:, np.newaxis, :]  # [ncrops, 1, 3]
        point_crops[..., :3] = point_crops[..., :3] - crop_centers
        point_crops[..., :3] = point_crops[..., :3] / crop_size * 2

    outputs = [point_crops, crop_indices]
    if return_shape: outputs += [idx_grid.shape[:3]]
    if return_crop_point_idxs:
        outputs += [crop_point_idxs]
    return tuple(outputs)


def sample_uniform_from_occupied_grids(v, crop_indices, xmin, crop_size, samples_per_grid=2048,
                                       overlap=True, dist_threshold=0.03):
    ncrops = crop_indices.shape[0]
    if overlap:
        crop_corners = xmin + crop_indices * 0.5 * crop_size  # [ncrops, 3]
    else:
        crop_corners = xmin + crop_indices * crop_size

    # Sample points uniformly from a grid
    uniform_samples = crop_corners[:, np.newaxis, :] + np.random.uniform(size=(ncrops, samples_per_grid, 3)) * crop_size
    uniform_samples = uniform_samples.reshape(-1, 3)
    tree = cKDTree(v, balanced_tree=False)
    dists, idxs = tree.query(uniform_samples, k=1, n_jobs=32)
    all_idxs = np.arange(uniform_samples.shape[0])
    target_idxs = all_idxs[dists > dist_threshold]
    target_samples = uniform_samples[target_idxs, :]
    target_samples = target_samples.astype(np.float32)

    return target_samples  # [?, 3]


def pca_normal_estimation(points, k=10):
    tree = cKDTree(points, balanced_tree=False)
    dists, idxs = tree.query(points, k=k + 1, n_jobs=32)
    idxs = idxs[:, 1:]
    npoints, k = idxs.shape
    neighbors = points[idxs.reshape(-1), :]
    neighbors = neighbors.reshape(npoints, k, 3)
    vectors = neighbors - points[:, np.newaxis, :]  # npoints*k*3
    vectors_trans = np.transpose(vectors, (0, 2, 1))  # npoints*3*k
    cov_matrix = np.matmul(vectors_trans, vectors)  # npoints*3*3
    u, s, v = np.linalg.svd(cov_matrix)
    est_normals = v[:, -1, :]  # npoints*3
    est_normals = est_normals / (np.linalg.norm(est_normals, axis=1, keepdims=True) + 1e-12)
    return est_normals


def occupancy_sparse_to_dense(occ_idx, grid_shape):
    dense = np.zeros(grid_shape, dtype=np.bool).ravel()
    occ_idx_f = (occ_idx[:, 0] * grid_shape[1] * grid_shape[2] + occ_idx[:, 1] * grid_shape[2] + occ_idx[:, 2])
    dense[occ_idx_f] = True
    dense = np.reshape(dense, grid_shape)
    return dense


def fit_sphere_through_points(points):
    points_mean = np.mean(points, axis=0, keepdims=True)
    N = points.shape[0]
    points_exp = np.tile(points[:, :, np.newaxis], [1, 1, 3])
    points_exp = points_exp.reshape(N, 9)
    delta = points - points_mean
    delta_exp = np.tile(delta, [1, 3])
    A = 2 * (points_exp * delta_exp).mean(axis=0).reshape(3, 3)

    points_squared = (points ** 2).sum(axis=1, keepdims=True)
    B = (points_squared * delta).mean(axis=0, keepdims=True).T

    AT_A = np.dot(A.T, A)
    AT_B = np.dot(A.T, B)
    try:
        center_T = np.dot(np.linalg.inv(AT_A), AT_B)
    except:
        return None, None
    center = center_T.T

    radius = np.sqrt(((points - center) ** 2).sum(axis=1).mean())

    return center, radius

def get_occupied_grid_fitting_sphere(v, xmin, xmax, crop_size, ntarget=2048, overlap=True,
                                     normalize_crops=False, return_shape=False, return_crop_point_idxs=False):
    point_crops, crop_indices, grid_shape, crop_point_idxs = np_get_occupied_idx(
        v=v,
        xmin=xmin,
        xmax=xmax,
        crop_size=crop_size,
        ntarget=ntarget,
        overlap=overlap,
        normalize_crops=normalize_crops,
        return_shape=True,
        return_crop_point_idxs=True
    )

    center_list = []
    radius_list = []
    valid_crop_idxs = []
    ncrops = len(crop_point_idxs)
    for n in range(ncrops):
        grid_points = v[crop_point_idxs[n]]

        if normalize_crops:
            # Normalize the grid points
            if overlap:
                grid_corner = xmin + crop_indices[n] * 0.5 * crop_size  # [3]
            else:
                grid_corner = xmin + crop_indices[n] * crop_size  # [3]
            grid_center = grid_corner + 0.5 * crop_size
            grid_points = (grid_points - grid_center) / (0.5 * crop_size)
        else:
            grid_points = grid_points - xmin

        center, radius = fit_sphere_through_points(grid_points)
        if center is not None:
            center_list.append(center)
            radius_list.append(radius)
            valid_crop_idxs.append(n)
    point_crops = point_crops[valid_crop_idxs, :, :]
    crop_indices = crop_indices[valid_crop_idxs, :]
    crop_point_idxs = [crop_point_idxs[idx] for idx in valid_crop_idxs]

    centers = np.concatenate(center_list).astype(np.float32)
    radius = np.array(radius_list).astype(np.float32)[:, np.newaxis]
    outputs = [point_crops, crop_indices, centers, radius]
    if return_shape: outputs += [grid_shape]
    if return_crop_point_idxs:
        outputs += [crop_point_idxs]
    return tuple(outputs)


def clip_radius_np(point_crops, centers, radius, min_radius=10.):
    clip_idxs = np.where(radius < min_radius)[0]
    clip_points = point_crops[clip_idxs, :, :]
    clip_points_centers = np.mean(clip_points, axis=1)  # gravity center
    clip_centers = centers[clip_idxs, :]  # x0
    vector = (clip_centers - clip_points_centers)
    vector_normed = vector / (np.linalg.norm(vector, axis=1, keepdims=True) + 1e-12)
    clip_points_new_centers = clip_points_centers + vector_normed * min_radius  # xc + r * (x0 - xc)
    centers[clip_idxs, :] = clip_points_new_centers
    radius[clip_idxs, :] = min_radius
    return centers, radius


def clip_radius_torch(point_crops, centers, radius, min_radius=10.):
    clip_idxs = torch.nonzero(radius < min_radius)[:, 0]
    clip_points = point_crops[clip_idxs, :, :]
    clip_points_centers = clip_points.mean(dim=1)  # gravity center
    clip_centers = centers[clip_idxs, :]  # x0
    vector = (clip_centers - clip_points_centers)
    vector_normed = vector / (torch.norm(vector, dim=1, keepdim=True) + 1e-12)
    clip_points_new_centers = clip_points_centers + vector_normed * min_radius  # xc + r * (x0 - xc)
    centers[clip_idxs, :] = clip_points_new_centers
    radius[clip_idxs, :] = min_radius
    return centers, radius


def sample_from_overlapping_area(crop_indices, xmin, crop_size, grid_shape, samples_per_grid=2048, overlap=True):
    ncrops = crop_indices.shape[0]
    if overlap:
        crop_corners = xmin + crop_indices * 0.5 * crop_size  # [ncrops, 3]
    else:
        crop_corners = xmin + crop_indices * crop_size

    # Sample points uniformly from a grid
    uniform_samples = crop_corners[:, np.newaxis, :] + np.random.uniform(size=(ncrops, samples_per_grid, 3)) * crop_size

    # Get neighbor indices of an occupied grid and mask out the neighboring grids without points or out of bound
    offset_x = np.array([0, 1])[:, np.newaxis, np.newaxis]
    offset_x = np.tile(offset_x, [1, 2, 2])
    offset_y = np.array([0, 1])[np.newaxis, :, np.newaxis]
    offset_y = np.tile(offset_y, [2, 1, 2])
    offset_z = np.array([0, 1])[np.newaxis, np.newaxis, :]
    offset_z = np.tile(offset_z, [2, 2, 1])
    neighbor_idx_offset = np.stack([offset_x, offset_y, offset_z], axis=3)
    neighbor_idx_offset = neighbor_idx_offset.reshape(-1, 3)
    neighbor_idxs = crop_indices[:, np.newaxis, :] + neighbor_idx_offset[np.newaxis, :, :] # [ncrops, nneighbors, 3]
    _, nneighbors, _ = neighbor_idxs.shape
    neighbor_idxs = neighbor_idxs.reshape(-1, 3)

    d, h, w = grid_shape
    neighbor_idxs_flatten = neighbor_idxs[:, 0] * (h * w) + neighbor_idxs[:, 1] * w + neighbor_idxs[:, 2]
    crop_indices_flatten = crop_indices[:, 0] * (h * w) + crop_indices[:, 1] * w + crop_indices[:, 2]
    mask = np.isin(neighbor_idxs_flatten, crop_indices_flatten) # [ncrops * nneighbors,]
    mask = mask.reshape(ncrops, nneighbors)

    # Mask out the grids with no neighbors containing points
    mask_with_neighbors = (mask.sum(axis=1) > 1)
    uniform_samples = uniform_samples[mask_with_neighbors, :, :]
    mask = mask[mask_with_neighbors, :].astype(np.float32)
    uniform_samples = uniform_samples.astype(np.float32)

    return uniform_samples, mask  # [?, samples_per_grid, 3] [?, 8]


def sample_from_grids_with_neighbors(crop_indices, xmin, crop_size, grid_shape, samples_per_grid=2048, overlap=True):
    d, h, w = grid_shape
    d_idxs = np.arange(d)
    h_idxs = np.arange(h)
    w_idxs = np.arange(w)
    dd, hh, ww = np.meshgrid(d_idxs, h_idxs, w_idxs)
    all_grid_idxs = np.stack([dd, hh, ww], axis=3)
    all_grid_idxs = all_grid_idxs.reshape(-1, 3)

    xx, yy, zz = np.meshgrid(*list(([0, 1],) * 3))
    neighbor_idx_offsets = np.stack([xx, yy, zz], axis=3)
    neighbor_idx_offsets = neighbor_idx_offsets.reshape(-1, 3)
    all_grid_neighbor_idxs = all_grid_idxs[:, np.newaxis, :] + neighbor_idx_offsets[np.newaxis, :, :] # [ngrids, 8, 3], ngrids==d*h*w

    # create neighbor mask for every crop grid idx
    ngrids, nneighbors, _ = all_grid_neighbor_idxs.shape
    all_grid_neighbor_idxs_flatten = all_grid_neighbor_idxs.reshape(-1, 3)
    all_grid_neighbor_idxs_flatten = all_grid_neighbor_idxs_flatten[:, 0] * (h * w) + all_grid_neighbor_idxs_flatten[:, 1] * w + all_grid_neighbor_idxs_flatten[:, 2]
    crop_indices_flatten = crop_indices[:, 0] * (h * w) + crop_indices[:, 1] * w + crop_indices[:, 2]
    mask_flatten = np.isin(all_grid_neighbor_idxs_flatten, crop_indices_flatten) # [ngrids * 8,]
    mask = mask_flatten.reshape(ngrids, nneighbors)
    mask_with_neighbor = np.any(mask, axis=1)

    grid_with_neighbor_idxs = all_grid_idxs[mask_with_neighbor, :] # [?, 3]
    grid_with_neighbor_corners = xmin + grid_with_neighbor_idxs * 0.5 * crop_size

    grid_with_neighbor_samples = grid_with_neighbor_corners[:, np.newaxis, :] + np.random.uniform(size=(grid_with_neighbor_corners.shape[0], samples_per_grid, 3)) * crop_size # [?, 3]
    grid_with_neighbor_samples = grid_with_neighbor_samples.reshape(-1, 3)
    grid_with_neighbor_samples = grid_with_neighbor_samples.astype(np.float32)

    return grid_with_neighbor_samples


def farthest_point_sample(xyz, nsamples): # bs*npoints*3
    centroids = np.zeros((xyz.shape[0], nsamples)).astype(np.long) # bs*nsamples
    distance = np.ones((xyz.shape[0], xyz.shape[1])).astype(np.float32) * 1e10 # bs*npoints
    farthest = np.random.randint(0, xyz.shape[1], size=(xyz.shape[0],)) # bs
    batch_idxs = np.arange(xyz.shape[0]) # bs
    for i in range(nsamples):
        centroids[:, i] = farthest
        centroid = np.expand_dims(xyz[batch_idxs, farthest, :], axis=1) # bs*1*3
        dist = ((xyz - centroid) ** 2).sum(axis=2) # bs*npoints
        mask = (dist < distance)
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=1)
    batch_idxs_exp = np.tile(batch_idxs[:, np.newaxis], [1, nsamples])
    query_points = xyz[batch_idxs_exp, centroids] # bs*nsamples*3
    return query_points # bs*nsamples*3


def random_point_sampling(xyz, nsamples): # bs*npoints*3
    bs, npoints, _ = xyz.shape
    rand_idxs = np.random.choice(npoints, size=(bs, nsamples), replace=False)
    batch_idxs = np.tile(np.arange(bs)[:, np.newaxis], [1, nsamples])
    rand_samples = xyz[batch_idxs, rand_idxs, :] # bs*nsamples*3
    return rand_samples # bs*nsamples*3

