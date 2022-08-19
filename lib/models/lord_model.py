import torch
import torch.nn as nn
from lib.models import decoder


class LoRD(nn.Module):
    def __init__(
        self,
        in_dim=4,
        motion_feat_dim=128
    ):
        super(LoRD, self).__init__()


        self.deform_net = decoder.ImNet(dim=in_dim, in_features=motion_feat_dim,
                                        num_filters=32, out_features=3)
        self.decoder = decoder.IGR(d_in=3 + motion_feat_dim, d_out=1,
                                   geometric_init=True, radius_init=0.05)


    def decode(self, x, c_s):
        bs, n_pts, n_neighbors, _ = x.shape
        decode_feat = torch.cat([x, c_s], dim=-1)
        logits = self.decoder(decode_feat)
        return logits

    def deform_points(self, x, t, c_m):
        if t == 0:
            return x
        else:
            bs, n_pts, n_neighbors, _ = x.shape
            deform_feat = torch.cat([x, t.reshape(bs, 1, 1, 1).repeat(1, n_pts, n_neighbors, 1), c_m], dim=-1)
            offset = self.deform_net(deform_feat)
            x = x + offset
            return x

    def forward(self, coords, t_val, c_m, c_s):
        new_coords = self.deform_points(coords, t_val, c_m)
        sdf = self.decode(new_coords, c_s)
        return sdf

