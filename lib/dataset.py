import os
import trimesh
import torch
import pickle
import numpy as np
from os.path import join
from torch.utils import data
from smplx import SMPL
from lib.utils import vec_normalize as normalize


class HumansDataset(data.Dataset):
    def __init__(self,
                 dataset_folder='dataset',
                 length_sequence=17,
                 num_optim_samples=10000):
        ''' Initialization of the human sequence dataset.
        '''

        # Attributes
        self.dataset_folder = dataset_folder
        self.length_sequence = length_sequence
        self.num_optim_samples = num_optim_samples

        bm_path = 'data/SMPL_NEUTRAL.pkl'
        self.bm = SMPL(model_path=bm_path)
        self.faces = np.load('data/smpl_faces.npy')
        self.beta = pickle.load(open('data/shapes.pkl', 'rb'))

        models_c = os.listdir(self.dataset_folder)
        self.models = [
            {'model': '_'.join(m.split('_')[:-1]),
             'start_idx': int(m.split('_')[-1])}
            for m in models_c
        ]

        self.part_file = 'data/train_800_parts.pt'

        self.n_parts = int(os.path.basename(self.part_file.split('_')[1]))


    def __len__(self):
        ''' 10000 batches for each epoch.
        '''
        return 10000


    def __getitem__(self, idx):
        model_idx = np.random.choice(len(self.models))
        model = self.models[model_idx]['model']
        start_idx = self.models[model_idx]['start_idx']

        data_dict = dict()

        data_dict_0 = self.get_data_dict(model, start_idx, time_step=0)
        for k, v in data_dict_0.items():
            k = k + '_0'
            data_dict[k] = v

        time_step = np.random.choice(self.length_sequence)
        data_dict_t = self.get_data_dict(model, start_idx, time_step=time_step)
        for k, v in data_dict_t.items():
            k = k + '_t'
            data_dict[k] = v

        data_dict['idx'] = model_idx
        data_dict['modelname'] = model

        return data_dict


    def get_data_dict(self, model, start_idx, time_step):
        data_dict = dict()
        time_val = torch.from_numpy(np.array(
            time_step / (self.length_sequence - 1), dtype=np.float32))
        rot_mtx, transl, bbox = self.get_transf(model, start_idx, time_step)

        '''dense point clouds'''
        pts_file = join(self.dataset_folder, f'{model}_{start_idx}',
                        'pcl_train', f'{time_step:04d}.npy')
        pts = np.load(pts_file)
        v = pts[:, :3]
        vn = pts[:, 3:]
        v_samples, vn_samples = self.random_point_sample(v, vn)

        data_dict['v'] = torch.from_numpy(v_samples).float()
        data_dict['vn'] = torch.from_numpy(vn_samples).float()
        data_dict['bbox'] = torch.from_numpy(bbox).float()
        data_dict['rot_mtx'] = torch.from_numpy(rot_mtx).float()
        data_dict['transl'] = torch.from_numpy(transl).float()
        data_dict['time_val'] = time_val
        data_dict['time_step'] = time_step

        return data_dict


    def get_model_dict(self, idx):
        return self.models[idx]


    def get_transf(self, model_name, start_idx, time_step):
        meta = torch.load(self.part_file)
        face_idx = meta['face_idx']
        alpha = meta['alpha']

        mesh = trimesh.load(join(self.dataset_folder,
                              f'{model_name}_{start_idx}', 'body_mesh',
                              f'{time_step:04d}.ply'), process=False)
        bbox = np.array(mesh.bounds).astype(np.float32)
        v = np.array(mesh.vertices[mesh.faces[face_idx]])
        transl = (alpha[:, :, None] * v).sum(axis=1)

        xx = normalize(v[:, 0] - transl)
        yy = normalize(mesh.face_normals[face_idx])
        zz = normalize(np.cross(xx, yy))
        rot_mtx = np.stack([xx, yy, zz], axis=-1)  # 1, 3, 3

        return rot_mtx, transl, bbox


    def random_point_sample(self, v, vn):
        idx = np.random.randint(v.shape[0], size=self.num_optim_samples)
        v_samples = v[idx, :]
        vn_samples = vn[idx, :]
        return v_samples, vn_samples

