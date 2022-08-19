import torch
import os
from os.path import join
import argparse
import yaml

from lib.models.lord_model import LoRD
from lib.models import decoder



def main(args):
    # Model
    model = LoRD(in_dim=cfg['model']['in_dim'],
                 motion_feat_dim=cfg['model']['latent_dim']).to(device)
    if args.texture:
        tex_model = decoder.TextureField(c_dim=cfg['model']['latent_dim'], dim=cfg['model']['in_dim']).to(device)
    else:
        tex_model = None

    ##########
    exp_name = args.exp_name
    ##########

    out_dir = cfg['training']['out_dir']
    data_dir = cfg['data']['dataset_folder']
    save_path = join(out_dir, exp_name, 'latent_vecs')
    os.makedirs(save_path, exist_ok=True)

    # load model parameters
    load_dict = torch.load(
        join(out_dir, 'checkpoints', 'model.pt')
    )
    model.load_state_dict(load_dict['model'])

    # you can append more sequences to the list
    seqs = [
        args.seq_name
    ]

    if not args.smpl_refine:
        from lib.reconstructor import Reconstructor
    else:
        from lib.reconstructor_refine_smpl import Reconstructor

    for p in seqs:
        p = {'model': '_'.join(p.split('_')[:-1]),
             'start_idx': int(p.split('_')[-1])}
        model_name = p['model']
        start_idx = p['start_idx']
        Reconstructor(model=model,
                      tex_model=tex_model,
                      model_name=model_name,
                      start_idx=start_idx,
                      data_folder=data_dir,
                      ply_folder=join(data_dir, f'{model_name}_{start_idx}', args.pcl_type),
                      version=f'{model_name}_{start_idx}.pt',
                      part_size=cfg['training']['part_size'],
                      device=device,
                      out_dir=out_dir,
                      save_path=save_path,
                      use_h4d_smpl=args.use_h4d_smpl
                      ).run()


if __name__ == '__main__':
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
    parser.add_argument('--smpl_refine', action='store_true', help='whether to use the inner body refinement')

    args = parser.parse_args()

    if args.smpl_refine:
        assert args.use_h4d_smpl

    with open(join('configs', args.config + '.yaml'), 'r') as f:
        cfg = yaml.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)