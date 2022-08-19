import torch
import os
from os.path import join
import argparse
import yaml

from torch.utils.data import DataLoader

from lib.models.lord_model import LoRD
from lib.trainer import Trainer
from lib.dataset import HumansDataset
from lib.utils import worker_init_fn


def main(cfg):
    # Model
    model = LoRD(in_dim=cfg['model']['in_dim'],
                 motion_feat_dim=cfg['model']['latent_dim']).to(device)
    out_dir = cfg['training']['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    dataset = HumansDataset(
        dataset_folder=cfg['data']['dataset_folder'],
        length_sequence=cfg['data']['length_sequence'],
        num_optim_samples=cfg['data']['training_samples']
    )
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4,
                              shuffle=True, worker_init_fn=worker_init_fn)

    num_seqs = len(dataset.models)
    num_parts = dataset.n_parts
    latent_size = cfg['model']['motion_feat_dim']

    init_std = 0.01
    c_m = torch.randn(num_seqs, num_parts, latent_size).type(torch.cuda.FloatTensor) * init_std
    c_s = torch.randn(num_seqs, num_parts, latent_size).type(torch.cuda.FloatTensor) * init_std
    c_m.requires_grad = True
    c_s.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": cfg['training']['lr'],
            },
            {
                "params": c_m,
                "lr": cfg['training']['lr'],
            },
            {
                "params": c_s,
                "lr": cfg['training']['lr'],
            },
        ]
    )

    save_path = join(out_dir, 'checkpoints')
    os.makedirs(save_path, exist_ok=True)

    Trainer(model=model, c_m=c_m, c_s=c_s, optimizer=optimizer,
            n_parts=num_parts, part_size=cfg['training']['part_size'],
            data_loader=train_loader,
            device=device, save_path=save_path).run()



if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a LoRD model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--g', type=str, default='0', help='gpu id')

    args = parser.parse_args()

    with open(join('configs', args.config + '.yaml'), 'r') as f:
        cfg = yaml.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(cfg)