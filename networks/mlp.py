import math

from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial

EMPTY = None
relu = partial(F.relu, inplace=True)  # saves a lot of memory


class ImplicitNet(nn.Module):
    """
    Represents a MLP;
    Original code from IGR
    """

    def __init__(
            self,
            d_in,
            dims,
            skip_in=(),
            d_out=4,
            geometric_init=True,
            radius_init=0.3,
            beta=0.0,
            output_init_gain=2.0,
            num_position_inputs=3,
            sdf_scale=1.0,
            dim_excludes_skip=False,
            combine_layer=1000,
            combine_type="average",
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param geometric_init if true, uses geometric initialization
               (to SDF of sphere)
        :param radius_init if geometric_init, then SDF sphere will have
               this radius
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        :param output_init_gain output layer normal std, only used for
                                output dimension >= 1, when d_out >= 1
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()

        dims = [d_in] + dims + [d_out]
        if dim_excludes_skip:
            for i in range(1, len(dims) - 1):
                if i in skip_in:
                    dims[i] += d_in

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.dims = dims
        self.combine_layer = combine_layer
        self.combine_type = combine_type

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            # if true preform geometric initialization
            if geometric_init:
                if layer == self.num_layers - 2:
                    # Note our geometric init is negated (compared to IDR)
                    # since we are using the opposite SDF convention:
                    # inside is +
                    nn.init.normal_(
                        lin.weight[0],
                        mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]) * sdf_scale,
                        std=0.00001,
                    )
                    nn.init.constant_(lin.bias[0], radius_init)
                    if d_out > 1:
                        # More than SDF output
                        nn.init.normal_(lin.weight[1:], mean=0.0, std=output_init_gain)
                        nn.init.constant_(lin.bias[1:], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                if d_in > num_position_inputs and (layer == 0 or layer in skip_in):
                    # Special handling for input to allow positional encoding
                    nn.init.constant_(lin.weight[:, -d_in + num_position_inputs:], 0.0)
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            # Vanilla ReLU
            self.activation = nn.ReLU()

    def forward(self, x, combine_inner_dims=(1,)):
        """
        :param x (..., d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        x_init = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer == self.combine_layer:
                x = utils.combine_interleaved(x, combine_inner_dims, self.combine_type)
                x_init = utils.combine_interleaved(
                    x_init, combine_inner_dims, self.combine_type
                )

            if layer < self.combine_layer and layer in self.skip_in:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            conf.get_list("dims"),
            skip_in=conf.get_list("skip_in"),
            beta=conf.get_float("beta", 0.0),
            dim_excludes_skip=conf.get_bool("dim_excludes_skip", False),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            **kwargs
        )


from torch import nn
import torch

#  import torch_scatter
import torch.autograd.profiler as profiler
import utils


# Resnet Blocks
class FCResBlock(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResMLP(nn.Module):
    def __init__(self, d_in, d_pc_latent, d_pix_latent, d_out, mlp_cfg=None):
        super().__init__()
        d_latent = d_pc_latent + d_pix_latent
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, mlp_cfg.d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(mlp_cfg.d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = mlp_cfg.n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = mlp_cfg.d_hidden

        self.combine_layer = mlp_cfg.combine_layer
        self.use_spade = mlp_cfg.use_spade

        self.blocks = nn.ModuleList(
            [FCResBlock(mlp_cfg.d_hidden, beta=mlp_cfg.beta) for i in range(mlp_cfg.n_blocks)]
        )

        if d_latent != 0:
            n_lin_z = min(mlp_cfg.combine_layer, mlp_cfg.n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, mlp_cfg.d_hidden) for i in range(n_lin_z)]
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, mlp_cfg.d_hidden) for _ in range(n_lin_z)]
                )
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if mlp_cfg.beta > 0:
            self.activation = nn.Softplus(beta=mlp_cfg.beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, zx, reshape_inner_dim=None, **kwargs):
        """
        :param zx (..., d_in + d_latent)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnetfc_infer"):
            # TODO, Structure of zx:
            ## xyz | pc_latent | pix_latent  (pcloudnerf)
            ## Or
            ## z_feature | latent | global_latent  (pixelnerf)
            ## Or
            ## z_feature | vox_latent  (voxelnerf)
            ## Or
            ## z_feature | vox_latent | pix_latent  (pixvox-nerf)
            assert zx.size(-1) == self.d_latent + self.d_in
            if self.d_latent > 0:
                x = zx[..., : self.d_in]
                z = zx[..., self.d_in:]
            else:
                x = zx
            if self.d_in > 0:
                x = self.lin_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            for blkid in range(self.n_blocks):
                if blkid == self.combine_layer:
                    # x = utils.combine_interleaved(
                    #     x, combine_inner_dims, self.combine_type
                    # )
                    x = x.reshape(-1, reshape_inner_dim, *x.shape[1:])  # (SB * B, latent) -> (SB, B, latent)

                if self.d_latent > 0 and blkid < self.combine_layer:
                    tz = self.lin_z[blkid](z)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    else:
                        x = x + tz

                x = self.blocks[blkid](x)
            out = self.lin_out(self.activation(x))
            return out


class ResMLPGRF(nn.Module):
    def __init__(self, d_in, d_pc_latent, d_pix_latent=0, d_out=4, mlp_cfg=None):
        super().__init__()
        self.d_in = d_in
        self.d_pc_latent = d_pc_latent
        self.d_pix_latent = d_pix_latent
        self.d_out = d_out

        # first stream: predict alpha from point-cloud feature
        self.alpha_mlp = ResMLP(d_in=0, d_latent=d_pc_latent, d_out=1, mlp_cfg=mlp_cfg)

        # second stream: predict rgb from both point-cloud feature and xyz (view direction)
        self.rgb_mlp = ResMLP(d_in=d_in, d_latent=d_pc_latent, d_out=3, mlp_cfg=mlp_cfg)

    def forward(self, zx, reshape_inner_dim=None):
        """
        :param zx (..., d_pc_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_pc_latent + self.d_in + self.d_pix_latent
            alpha = self.alpha_mlp(zx[..., : self.d_pc_latent], reshape_inner_dim)
            rgb = self.rgb_mlp(zx, reshape_inner_dim)

            return torch.cat([rgb, alpha], dim=-1)


class ResMLPDisent(nn.Module):
    def __init__(self, d_in, d_pc_latent, d_pix_latent=0, d_out=4, mlp_cfg=None):
        super().__init__()
        self.d_in = d_in
        self.d_pc_latent = d_pc_latent
        self.d_pix_latent = d_pix_latent
        self.d_out = d_out

        # first stream: predict alpha from point-cloud feature and xyz
        self.alpha_mlp = ResMLP(d_in, d_pc_latent, 0, d_out=1, mlp_cfg=mlp_cfg)

        # second stream: predict rgb from [point-cloud feature, pix features, xyz (view direction)]
        self.rgb_mlp = ResMLP(d_in, d_pc_latent, d_pix_latent, d_out=3, mlp_cfg=mlp_cfg)

    def forward(self, zx, reshape_inner_dim=None):
        """
        :param zx (..., d_pc_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_pc_latent + self.d_in + self.d_pix_latent
            alpha = self.alpha_mlp(zx[..., :self.d_in + self.d_pc_latent], reshape_inner_dim)
            rgb = self.rgb_mlp(zx, reshape_inner_dim)

            return torch.cat([rgb, alpha], dim=-1)


class MLPGRAF(nn.Module):
    def __init__(self, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, mlp_cfg=None):
        """
        """
        super(MLPGRAF, self).__init__()
        D = mlp_cfg.n_blocks
        W = mlp_cfg.d_hidden

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, **kwargs):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class VisibilityNet(nn.Module):
    def __init__(self, d_in, d_hidden, beta=0, n_blocks=5):
        super().__init__()
        self.lin_in = nn.Linear(d_in, d_hidden)
        nn.init.constant_(self.lin_in.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.blocks = nn.ModuleList(
            [FCResBlock(d_hidden, beta=beta) for _ in range(n_blocks)]
        )

        self.lin_out = nn.Linear(d_hidden, 1)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.lin_in(x)
        for blk in self.blocks:
            x = blk(x)

        out = self.lin_out(self.activation(x))
        return out


################################################################
# siren layer
################################################################

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma=None, beta=None):
        out = F.linear(x, self.weight, self.bias)

        # FiLM modulation

        if gamma is not None:
            out = out * gamma

        if beta is not None:
            out = out + beta

        out = self.activation(out)
        return out


# mapping network

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=0.1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class MappingNetwork(nn.Module):
    def __init__(self, *, dim, dim_out, depth=3, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), nn.LeakyReLU(0.2)])

        self.net = nn.Sequential(*layers)

        self.to_gamma = nn.Linear(dim, dim_out)
        self.to_beta = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)


# siren network

class SirenBackbone(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1., w0_initial=30., use_bias=True,
                 final_activation=None):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias,
                                     activation=final_activation)

    def forward(self, x, gamma, beta):
        for layer in self.layers:
            x = layer(x, gamma, beta)
        return self.last_layer(x)


# generator

class SIREN(nn.Module):
    def __init__(self, d_in, d_pc_latent, d_pix_latent, d_out, mlp_cfg=None):
        super().__init__()

        d_latent = d_pc_latent + d_pix_latent
        self.n_blocks = mlp_cfg.n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = mlp_cfg.d_hidden

        self.mapping = MappingNetwork(
            dim=d_latent,
            dim_out=mlp_cfg.d_hidden
        )

        self.siren = SirenBackbone(
            dim_in=3,
            dim_hidden=mlp_cfg.d_hidden,
            dim_out=mlp_cfg.d_hidden,
            num_layers=mlp_cfg.n_blocks
        )

        self.to_alpha = nn.Linear(mlp_cfg.d_hidden, 1)

        self.to_rgb_siren = SirenLayer(
            dim_in=mlp_cfg.d_hidden + 3,
            dim_out=mlp_cfg.d_hidden
        )

        self.to_rgb = nn.Linear(mlp_cfg.d_hidden, 3)

    def forward(self, zx, **kwargs):
        latent = zx[..., :self.d_latent]
        coor = zx[..., self.d_latent:self.d_latent + 3]
        viewdir = zx[..., self.d_latent + 3:]

        gamma, beta = self.mapping(latent)

        x = self.siren(coor, gamma, beta)
        alpha = self.to_alpha(x)

        x = self.to_rgb_siren(torch.cat([x, viewdir], dim=1), gamma, beta)
        rgb = self.to_rgb(x)
        out = torch.cat((rgb, alpha), dim=-1)

        return out
