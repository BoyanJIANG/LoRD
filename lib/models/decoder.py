import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.models.layers import ResnetBlockFC


class TextureField(nn.Module):
    def __init__(self, c_dim=128, z_dim=0, dim=3,
                 hidden_size=128, leaky=True):
        super().__init__()
        self.c_dim = c_dim

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)


        self.fc_p = nn.Linear(dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)
        # self.block1 = ResnetBlockPointwise(
        #     hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        # self.block2 = ResnetBlockPointwise(
        #     hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        # self.block3 = ResnetBlockPointwise(
        #     hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        # self.block4 = ResnetBlockPointwise(
        #     hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

        self.fc_cz_0 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_1 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_2 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_3 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_4 = nn.Linear(c_dim + z_dim, hidden_size)

        self.fc_out = nn.Linear(hidden_size, 3)


        # Initialization
        nn.init.zeros_(self.fc_out.weight)

    def forward(self, p, cz):

        net = self.fc_p(p)
        net = net + self.fc_cz_0(cz)
        net = self.block0(net)
        net = net + self.fc_cz_1(cz)
        net = self.block1(net)
        net = net + self.fc_cz_2(cz)
        net = self.block2(net)
        net = net + self.fc_cz_3(cz)
        net = self.block3(net)
        net = net + self.fc_cz_4(cz)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = torch.sigmoid(out)

        return out


class ImNet(nn.Module):
    """ImNet layer py-torch implementation."""

    def __init__(self, dim=3, in_features=32, out_features=1, num_filters=32, activation=nn.LeakyReLU(0.2)):
        """Initialization.
        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          num_filters: int, width of the second to last layer.
          activation: activation function.
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.num_filters = num_filters
        self.activ = activation
        self.fc0 = nn.Linear(self.dimz, num_filters * 16)
        self.fc1 = nn.Linear(self.dimz + num_filters * 16, num_filters * 8)
        self.fc2 = nn.Linear(self.dimz + num_filters * 8, num_filters * 4)
        self.fc3 = nn.Linear(self.dimz + num_filters * 4, num_filters * 2)
        self.fc4 = nn.Linear(self.dimz + num_filters * 2, num_filters * 1)
        self.fc5 = nn.Linear(num_filters * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    def forward(self, x):
        """Forward method.
        Args:
          x: `[batch_size, n_pts, dim+in_features]` tensor, inputs to decode.
        Returns:
          x_: output through this layer.
        """
        x_ = x
        for dense in self.fc[:4]:
            x_ = self.activ(dense(x_))
            x_ = torch.cat([x_, x], dim=-1)
        x_ = self.activ(self.fc4(x_))
        x_ = self.fc5(x_)
        # x_ = x_.squeeze(-1)
        return x_



class IGR(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        dims=[256,]*6,
        skip_in=[3],
        geometric_init=True,
        radius_init=1,
        beta=100
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
