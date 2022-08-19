import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
from os.path import join
from lib.evaluator import MeshEvaluator


#################################################################
generation_dir = 'out/lord/fit_sparse_pcl_h4d_pose'
pred_path = os.path.join(generation_dir, 'vis')
gt_path = 'dataset'
#################################################################

out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')

evaluator = MeshEvaluator(n_points=100000)

models = [
    '03284_shortlong_simple_87'
]

eval_dicts = []
pred_mesh = []
idx = 0
all_results_dict = {
            'completeness': [],
            'accuracy': [],
            'normals completeness': [],
            'normals accuracy': [],
            'normals consistency': [],
            'completeness2': [],
            'accuracy2': [],
            'chamfer-L2': [],
            'chamfer-L1': [],
            'fscore': [],
        }

for m in tqdm(models, ncols=80):
    idx += 1
    modelname = '_'.join(m.split('_')[:-1])
    start_idx = int(m.split('_')[-1])
    eval_dict = {
        'idx': idx,
        'class name': 'n/a',
        'modelname': modelname,
    }
    eval_dicts.append(eval_dict)
    pred_mesh_dir = join(pred_path, f'{modelname}_{start_idx}', '0.007')
    gt_mesh_dir = join(gt_path, f'{modelname}_{start_idx}', 'mesh')

    for i in range(17):
        mesh_pred = trimesh.load(join(pred_mesh_dir, f'{i:04d}.ply'), process=False)
        mesh_gt = trimesh.load(join(gt_mesh_dir, f'{i:04d}.ply'), process=False)


        eval_dict_mesh = evaluator.eval_mesh(mesh_pred, mesh_gt)
        for k, v in eval_dict_mesh.items():
            eval_dict['%s %d (mesh)' % (k, i)] = v

        for k, v in eval_dict_mesh.items():
            all_results_dict[k].append(v)

    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean()
    eval_df_class.to_csv(out_file_class)

    # Print results
    eval_df_class.loc['mean'] = eval_df_class.mean()
    print(eval_df_class)

log_str = f'Total Seqs: {idx}'
for k, v in all_results_dict.items():
    log_str += f' | {k}: {np.mean(v):g}'
print(log_str)
with open(os.path.join(generation_dir, 'eval_results.txt'), 'w') as f:
    f.write(log_str)
