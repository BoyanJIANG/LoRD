from os.path import join
import json
import numpy as np
import torch
import trimesh
from smplx import SMPL


part_radius = 0.05
n_parts = 2127  # number of the sampled parts
overlap_ratio = 0.5  # larger for more parts

# Construct template mesh
model_path = 'data/SMPL_NEUTRAL.pkl'
bm = SMPL(model_path=model_path)
faces = np.load('data/smpl_faces.npy')
poses = torch.zeros([1, 72])
# Adjust hip joints to prevent sampled parts from including region of both legs
poses[:, 3:6] = torch.tensor([0, 0, 0.5])
poses[:, 6:9] = torch.tensor([0, 0, -0.5])
template_v = bm(body_pose=poses[:, 3:],
                global_orient=poses[:, :3]).vertices[0]
template_mesh = trimesh.Trimesh(template_v.detach().cpu().numpy(), faces, process=False)

# Select face indices of fingers and toes for sampling extra parts
smpl_part_idx = json.load(open('data/smpl_vert_segmentation.json'))
extra_vert_idx = list(set(smpl_part_idx['leftToeBase'] + smpl_part_idx['rightToeBase'] +
                          smpl_part_idx['leftHandIndex1'] + smpl_part_idx['rightHandIndex1']))
extra_face_idx = []
for v in extra_vert_idx:
    extra_face_idx.extend(template_mesh.vertex_faces[v])
extra_face_idx = list(set(extra_face_idx))
extra_face_idx.remove(-1)
extra_face_idx = np.random.permutation(extra_face_idx)

n_uniform_samples = 1000000
_, uniform_face_idx = template_mesh.sample(n_uniform_samples, return_index=True)
uniform_face_idx = np.random.permutation(uniform_face_idx)
alpha = np.random.dirichlet((1,) * 3, n_uniform_samples)

# Iteratively sample the part center and remove its involved points
tm_v = template_mesh.vertices[template_mesh.faces[uniform_face_idx]]
tm_points = (alpha[:, :, None] * tm_v).sum(axis=1)
rest_idx = np.arange(n_uniform_samples)
choose_idx = []
while rest_idx.size > 0:
    idx = np.random.choice(rest_idx)
    choose_idx.append(idx)
    dist = np.linalg.norm(tm_points[rest_idx] - tm_points[[idx]], axis=-1)
    threshold = part_radius * (1 - overlap_ratio)  # parts are overlapping
    rest_idx = rest_idx[dist >= threshold]
face_idx = uniform_face_idx[choose_idx]
alpha = alpha[choose_idx]

# Sample more parts in the finger and toe area
n_extra_parts = n_parts - len(choose_idx)
assert n_extra_parts > 0, 'You should increase the number of part samples or decrease the overlap ratio.'
choose_extra_face_idx = np.random.choice(extra_face_idx, size=n_extra_parts, replace=False)
face_idx = np.hstack([face_idx, choose_extra_face_idx])
alpha = np.vstack([alpha, np.random.dirichlet((1,) * 3, n_extra_parts)])

torch.save({'face_idx': face_idx,
            'alpha': alpha,
            'n_uniform': len(choose_idx),
            'n_extra': n_extra_parts},
            join(f'data/{n_parts}_parts.pt'))



