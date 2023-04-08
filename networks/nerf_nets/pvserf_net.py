from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from networks.encoders import pix2vox
from networks.encoders import encoder
from networks.code import PositionalEncoding
from networks import mlp
from networks.encoders.pointnet2_part_seg_msg import PointNet2PropEncoder
from networks.graphx_conv.pix2pc_world import Pixel2PointcloudWorld
from utils import repeat_interleave


class PVSeRFNet(torch.nn.Module):
    '''
    1. pc_generator -> world pc
    2. input world xyz into nerf MLP
    '''

    def __init__(self, model_cfg):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        ##########################################
        # Pixel-aligned
        ##########################################
        self.encoder = getattr(encoder, f'{model_cfg.encoder.type}Encoder')(model_cfg.encoder)
        self.use_encoder = model_cfg.use_encoder  # Image features?

        ##########################################
        # Voxel-aligned
        ##########################################
        self.use_voxel_encoder = model_cfg.use_voxel_encoder
        self.freeze_voxel_encoder = model_cfg.freeze_voxel_encoder  # Stop point-feature extractor gradient (freeze weights)
        if self.use_voxel_encoder:
            self.voxel_encoder = getattr(pix2vox, model_cfg.voxel_encoder.name)(model_cfg.voxel_encoder)
            if model_cfg.vox_encoder_ckpt:
                print('INFO: Loading voxel encoder weights...')
                state_dict = torch.load(model_cfg.vox_encoder_ckpt)
                rename_dict = lambda dict: OrderedDict([(key.split("module.")[-1], dict[key]) for key in dict])

                self.voxel_encoder.encoder.load_state_dict(rename_dict(state_dict['encoder_state_dict']))
                self.voxel_encoder.decoder.load_state_dict(rename_dict(state_dict['decoder_state_dict']))

            self.voxel_latent_size = model_cfg.voxel_encoder.latent_size
            self.multi_scale_query = model_cfg.voxel_encoder.ms_query

            if self.freeze_voxel_encoder:
                print('INFO: voxel encoder frozen')

        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * model_cfg.voxel_encoder.displacment
                displacments.append(input)
        self.register_buffer("displacments", torch.Tensor(displacments))

        ##########################################
        # pcloud-aligned
        ##########################################
        self.pc_generator = Pixel2PointcloudWorld()
        if model_cfg.graphx_ckpt:
            print('INFO: Loading pointcloud generator weights...')
            self.pc_generator.load_state_dict(torch.load(model_cfg.graphx_ckpt)['model_state_dict'])
        self.freeze_graphx = model_cfg.freeze_graphx
        if self.freeze_graphx:
            print('INFO: GraphX-Conv pc_generator frozen')

        self.pc_encoder = PointNet2PropEncoder(normal_channel=False)
        self.freeze_pc_encoder = model_cfg.freeze_pc_encoder  # Stop point-feature extractor gradient (freeze weights)
        if self.freeze_pc_encoder:
            print('INFO: pc encoder frozen')
        if model_cfg.pc_encoder_ckpt:
            print('INFO: Loading encoder weights...')
            self.pc_encoder.load_state_dict(torch.load(model_cfg.pc_encoder_ckpt)['model_state_dict'])

        self.use_xyz = model_cfg.use_xyz

        assert self.use_voxel_encoder or self.use_encoder or self.use_xyz  # Must use some feature..

        self.nearest_K = model_cfg.nearest_K
        self.use_KNN = self.nearest_K is not None
        if self.use_KNN:
            print(f'INFO: using {self.nearest_K} nearest points to get features')

        self.use_softmax = model_cfg.use_softmax

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = model_cfg.normalize_z

        self.stop_encoder_grad = model_cfg.stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        if self.stop_encoder_grad:
            print('INFO: Encoder frozen')
        else:
            print('INFO: joint train encoder')

        self.use_code = model_cfg.use_code  # Positional encoding
        self.use_code_viewdirs = model_cfg.use_code_viewdirs  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = model_cfg.use_viewdirs

        d_vox_latent = self.voxel_latent_size if self.use_voxel_encoder else 0
        d_pc_latent = self.pc_encoder.latent_size
        d_pix_latent = self.encoder.latent_size if self.use_encoder else 0

        d_in = 3

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(model_cfg.code, d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        d_out = 4

        self.mlp_coarse = getattr(mlp, model_cfg.mlp_coarse.name)(d_in, d_vox_latent + d_pc_latent, d_pix_latent, d_out,
                                                                  model_cfg.mlp_coarse)
        self.mlp_fine = getattr(mlp, model_cfg.mlp_fine.name)(d_in, d_vox_latent + d_pc_latent, d_pix_latent, d_out,
                                                              model_cfg.mlp_fine)

        assert model_cfg.mlp_coarse.beta == model_cfg.mlp_fine.beta
        self.sigma_act = torch.relu if model_cfg.mlp_fine.beta == 0 else F.softplus

        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_vox_latent = d_vox_latent
        self.d_pix_latent = d_pix_latent
        self.d_pc_latent = d_pc_latent

        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.register_buffer('rot_about_z_inv', torch.tensor(
            [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32
        ))
        self.register_buffer('rot_about_y_inv', torch.tensor(
            [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32
        ))

        self.register_buffer('rot_about_z', torch.tensor(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32
        ))
        self.register_buffer('rot_about_y', torch.tensor(
            [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32
        ))

        self.register_buffer('rgb2gray_filter', torch.tensor(
            [0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1))

    def encode(self, src_images, poses, init_pclouds, focal, c=None, extrinsic=None, intrinsic=None):
        """
        :param src_depths (B, NS, H, W)
        NS is number of input (aka source or reference) views
        :param src_extrin_invs (B, NS, 4, 4)
        :param src_intrin_invs (B, NS, 4, 4)
        :param src_images (SB, NS, 3, H, W)
        """
        ##########################################
        # Pixel-aligned
        ##########################################
        self.num_objs = src_images.size(0)
        if len(src_images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == src_images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = src_images.size(1)
            src_images_tile = src_images.reshape(-1, *src_images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(src_images_tile)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = src_images_tile.shape[-1]
        self.image_shape[1] = src_images_tile.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        ##########################################
        # Voxel-aligned
        ##########################################
        # encode voxel features
        self.voxel_features = None

        raw_features, gen_voxels = self.voxel_encoder(src_images)
        # (batch_size, n_views, 9, 32, 32, 32) > (batch_size, 9, 32, 32, 32): only 1 view, so del dim1
        self.voxel_features = raw_features.squeeze(1)

        ##########################################
        # pcloud-aligned
        ##########################################
        extrinsic = extrinsic.reshape(-1, 4, 4)
        intrinsic = intrinsic.reshape(-1, 4, 4)
        gray_images = F.conv2d(src_images_tile, self.rgb2gray_filter)  # (SB, 1, H, W)
        pred_pc_world = self.pc_generator(gray_images, init_pclouds, extrinsic, intrinsic)  # (SB, N, 3)
        if self.freeze_graphx:
            pred_pc_world = pred_pc_world.detach()

        # rotate pclouds
        pred_pc_world = pred_pc_world.permute(0, 2, 1)  # (SB, N, 3) -> (SB, 3, N)
        rot_pred_pc_world = torch.cat([pred_pc_world, torch.ones_like(pred_pc_world[:, 0:1, :])], dim=1)  # (SB, 4, N)
        rot_pred_pc_world = self.rot_about_y[None] @ self.rot_about_z[None] @ rot_pred_pc_world

        # pclouds features
        self.pred_pc_world = rot_pred_pc_world[:, :3]  # (SB, 3, N)
        self.pc_features = self.pc_encoder(self.pred_pc_world)  # (SB, C, N)

        ret_dict = {
            'pc_world': pred_pc_world,  # (SB, 3, N)
            'voxel': gen_voxels  # torch.Size([batch_size, n_views, 32, 32, 32])
        }
        return ret_dict

    def agg_feats(self, point_wise_dist, feats):
        '''
        input are src_pclouds, xyz, and pc_features
        src_pclouds: (SB, 3, N)
        xyz: (SB, B, 3)
        pc_features: (SB, C, N)
        output latent: (SB, B, C)

        First correlate xyz and src_pclouds, get (SB, B, N).
        Then conduct softmax at column dim (get soft attention mask).
        At last, calculate attenuated pc_features (permuted) using mask, get (SB, B, C)
        '''
        if not self.use_KNN:
            if self.use_softmax:
                latent = F.softmax(
                    # torch.exp(-torch.cdist(xyz, self.src_pclouds.permute(0, 2, 1))),  # use 1/exp(dist)
                    # torch.cdist(xyz, self.src_pclouds.permute(0, 2, 1)), # use dist
                    -point_wise_dist,  # use dist (0325 reproduce)
                    dim=-1
                ) @ feats.permute(0, 2, 1)  # no detach (0325)
            else:
                point_wise_dist = point_wise_dist / np.sqrt(12)  # divided by max dist
                latent = point_wise_dist.detach() @ feats.permute(0, 2, 1)
        else:
            sort_inds = torch.argsort(point_wise_dist, dim=-1)
            KNN_inds = sort_inds[..., :self.nearest_K]  # K points with the smallest dist
            KNN_dist = torch.gather(point_wise_dist, -1, KNN_inds)
            mask = torch.zeros_like(point_wise_dist)
            if self.use_softmax:
                mask.scatter_(-1, KNN_inds, F.softmax(-KNN_dist, dim=-1))
            else:
                # KNN_dist = KNN_dist / np.sqrt(12)  # divided by max dist
                # KNN_dist = torch.exp(-KNN_dist)  # use exp(-dist)
                # KNN_dist = 1/ (1 + torch.exp(KNN_dist))  # use sigmoid
                # KNN_dist = 1 / torch.tanh(KNN_dist)
                # KNN_dist = (KNN_dist - KNN_dist.mean()) / KNN_dist.std()
                # KNN_dist = torch.sigmoid(-KNN_dist)
                # KNN_dist = torch.abs(F.logsigmoid(KNN_dist))
                # KNN_dist = F.softmin(KNN_dist)  # not injective
                # KNN_dist = F.softplus(-KNN_dist)
                # KNN_dist = F.softsign(-KNN_dist)
                KNN_dist = F.softsign(KNN_dist)
                # KNN_dist = 1 / (1 + KNN_dist)
                mask.scatter_(-1, KNN_inds, KNN_dist)

            latent = mask.detach() @ feats.permute(0, 2, 1)

        return latent

    def forward(self, xyz_world, coarse=True, viewdirs=None):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz_world.shape
            NS = self.num_views_per_obj

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz_world, NS)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                # * Encode the xyz coordinates
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul(
                        self.poses[:, None, :3, :3], viewdirs
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat(
                        (z_feature, viewdirs), dim=1
                    )  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            # TODO: Order matters:
            ## xyz | vox_latent | pix_latent | pc_latent

            ##########################################
            # Voxel-aligned
            ##########################################
            if self.use_voxel_encoder:
                # xyz_world is in world, rotate it to align with GT mesh(voxel)
                xyz_homo = torch.cat([xyz_world.transpose(1, 2), torch.ones(SB, 1, B).to(xyz_world)], dim=1)  # (SB, 4, B)
                xyz_rot = self.rot_about_y_inv @ self.rot_about_z_inv @ xyz_homo
                xyz_vox = xyz_rot[:, :3].transpose(1, 2)  # (SB, B, 3)

                xyz_vox = xyz_vox[:, None, None, :, :]  # (SB, 1, 1, B, 3)

                if not self.multi_scale_query:
                    # directly query feature from voxel feature grid
                    latent = F.grid_sample(self.voxel_features, xyz_vox, padding_mode='zeros',
                                           align_corners=True)  # (SB, C, 1, 1, B)
                    latent = latent.transpose(1, 4).reshape(-1, self.d_vox_latent)  # (SB * B, C)
                else:
                    # From Geo-PIFu (multi-scale feature query)
                    # -----------------------------------------------------
                    # add displacements
                    xyz_vox = torch.cat([xyz_vox + d for d in self.displacments], dim=2)  # (B,1,7,B,3)

                    # grid sampling
                    latent = torch.nn.functional.grid_sample(self.voxel_features, xyz_vox, padding_mode='zeros',
                                                             align_corners=True)  # (SB,C=8,1,7,B)
                    latent = latent.transpose(1, 4).reshape(SB, B, -1)  # (SB, B, (7*C)=56)
                    latent = latent.reshape(-1, self.d_vox_latent)  # (SB * B, (7*C)=56)

                    # -----------------------------------------------------

                if self.freeze_voxel_encoder:
                    latent = latent.detach()

                mlp_input = torch.cat((mlp_input, latent), dim=-1)

            ##########################################
            # Pixel-aligned
            ##########################################
            if self.use_encoder:
                NS = self.num_views_per_obj

                # Grab encoder's latent code.
                uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
                uv *= repeat_interleave(
                    self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
                )
                uv += repeat_interleave(
                    self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
                )  # (SB * NS=1, B, 2)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB * NS=1, latent, B)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(
                    -1, self.d_pix_latent
                )  # (SB * NS=1 * B, latent)

                mlp_input = torch.cat((mlp_input, latent), dim=-1)

            ##########################################
            # pcloud-aligned
            ##########################################
            # add pc-aligned features
            # with torch.no_grad():  # no_grad (0513)
            #     dist = torch.cdist(xyz, self.src_pclouds.permute(0, 2, 1))
            # xyz_world is in world space, so is pred_pc_world
            dist = torch.cdist(xyz_world, self.pred_pc_world.permute(0, 2, 1))  # (0325)
            latent = self.agg_feats(dist, self.pc_features)
            latent = latent.reshape(-1, self.d_pc_latent)  # C is 128, (SB * B, C)

            if self.freeze_pc_encoder:
                latent = latent.detach()

            mlp_input = torch.cat((mlp_input, latent), dim=-1)

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    reshape_inner_dim=B
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    reshape_inner_dim=B
                )

            # Interpret the output
            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), self.sigma_act(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)
        return output


class PVSeRFSRNNet(PVSeRFNet):
    '''
    1. pc_generator -> world pc
    2. input world xyz into nerf MLP
    '''

    def __init__(self, model_cfg):
        """
        :param conf PyHocon config subtree 'model'
        """
        super(PVSeRFNet, self).__init__()  # rewrite init
        ##########################################
        # Pixel-aligned
        ##########################################
        self.encoder = getattr(encoder, f'{model_cfg.encoder.type}Encoder')(model_cfg.encoder)
        self.use_encoder = model_cfg.use_encoder  # Image features?

        ##########################################
        # Voxel-aligned
        ##########################################
        self.use_voxel_encoder = model_cfg.use_voxel_encoder
        self.freeze_voxel_encoder = model_cfg.freeze_voxel_encoder  # Stop point-feature extractor gradient (freeze weights)
        if self.use_voxel_encoder:
            self.voxel_encoder = getattr(pix2vox, model_cfg.voxel_encoder.name)(model_cfg.voxel_encoder)
            if model_cfg.vox_encoder_ckpt:
                print('INFO: Loading encoder weights...')
                state_dict = torch.load(model_cfg.vox_encoder_ckpt)
                rename_dict = lambda dict: OrderedDict([(key.split("module.")[-1], dict[key]) for key in dict])

                self.voxel_encoder.encoder.load_state_dict(rename_dict(state_dict['encoder_state_dict']))
                self.voxel_encoder.decoder.load_state_dict(rename_dict(state_dict['decoder_state_dict']))

            self.voxel_latent_size = model_cfg.voxel_encoder.latent_size
            self.multi_scale_query = model_cfg.voxel_encoder.ms_query

            if self.freeze_voxel_encoder:
                print('INFO: voxel encoder frozen')

        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * model_cfg.voxel_encoder.displacment
                displacments.append(input)
        self.register_buffer("displacments", torch.Tensor(displacments))

        ##########################################
        # pcloud-aligned
        ##########################################
        self.pc_generator = Pixel2PointcloudWorld(is_srn_proj=True)
        if model_cfg.graphx_ckpt:
            print('INFO: Loading pointcloud generator weights...')
            self.pc_generator.load_state_dict(torch.load(model_cfg.graphx_ckpt)['model_state_dict'])
        self.freeze_graphx = model_cfg.freeze_graphx
        if self.freeze_graphx:
            print('INFO: GraphX-Conv pc_generator frozen')

        self.pc_encoder = PointNet2PropEncoder(normal_channel=False)
        self.freeze_pc_encoder = model_cfg.freeze_pc_encoder  # Stop point-feature extractor gradient (freeze weights)
        if self.freeze_pc_encoder:
            print('INFO: pc encoder frozen')
        if model_cfg.pc_encoder_ckpt:
            print('INFO: Loading encoder weights...')
            self.pc_encoder.load_state_dict(torch.load(model_cfg.pc_encoder_ckpt)['model_state_dict'])

        self.use_xyz = model_cfg.use_xyz

        assert self.use_voxel_encoder or self.use_encoder or self.use_xyz  # Must use some feature..

        self.nearest_K = model_cfg.nearest_K
        self.use_KNN = self.nearest_K is not None
        if self.use_KNN:
            print(f'INFO: using {self.nearest_K} nearest points to get features')

        self.use_softmax = model_cfg.use_softmax

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = model_cfg.normalize_z

        self.stop_encoder_grad = model_cfg.stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        if self.stop_encoder_grad:
            print('INFO: Encoder frozen')
        else:
            print('INFO: joint train encoder')

        self.use_code = model_cfg.use_code  # Positional encoding
        self.use_code_viewdirs = model_cfg.use_code_viewdirs  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = model_cfg.use_viewdirs

        d_vox_latent = self.voxel_latent_size if self.use_voxel_encoder else 0
        d_pc_latent = self.pc_encoder.latent_size
        d_pix_latent = self.encoder.latent_size if self.use_encoder else 0

        d_in = 3

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(model_cfg.code, d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        d_out = 4

        self.mlp_coarse = getattr(mlp, model_cfg.mlp_coarse.name)(d_in, d_vox_latent + d_pc_latent, d_pix_latent, d_out,
                                                                  model_cfg.mlp_coarse)
        self.mlp_fine = getattr(mlp, model_cfg.mlp_fine.name)(d_in, d_vox_latent + d_pc_latent, d_pix_latent, d_out,
                                                              model_cfg.mlp_fine)

        assert model_cfg.mlp_coarse.beta == model_cfg.mlp_fine.beta
        self.sigma_act = torch.relu if model_cfg.mlp_fine.beta == 0 else F.softplus

        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_vox_latent = d_vox_latent
        self.d_pix_latent = d_pix_latent
        self.d_pc_latent = d_pc_latent

        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        # for SRN, GT pcloud should rot about z-axis for -pi/2
        self.register_buffer('pc_rot_about_z', torch.tensor(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=torch.float32
        ))

        self.register_buffer('voxel_R_inv', torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            dtype=torch.float32
        ))

        self.register_buffer('rgb2gray_filter', torch.tensor(
            [0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1))

    def encode(self, src_images, poses, init_pclouds, focal, c=None, extrinsic=None, intrinsic=None):
        """
        :param src_depths (B, NS, H, W)
        NS is number of input (aka source or reference) views
        :param src_extrin_invs (B, NS, 4, 4)
        :param src_intrin_invs (B, NS, 4, 4)
        :param src_images (SB, NS, 3, H, W)
        """
        ##########################################
        # Pixel-aligned
        ##########################################
        self.num_objs = src_images.size(0)
        assert len(src_images.shape) == 5  # src_images must be (B, NS, 3, H, W), even B=1,NS=1
        if len(src_images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == src_images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = src_images.size(1)
            src_images_tile = src_images.reshape(-1, *src_images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(src_images_tile)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = src_images_tile.shape[-1]
        self.image_shape[1] = src_images_tile.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        ##########################################
        # Voxel-aligned
        ##########################################
        # encode voxel features
        self.voxel_features = None

        raw_features, gen_voxels = self.voxel_encoder(src_images)
        # (batch_size, n_views, 9, 32, 32, 32) > (batch_size, 9, 32, 32, 32): only 1 view, so del dim1
        self.voxel_features = raw_features.squeeze(1)

        ##########################################
        # pcloud-aligned
        ##########################################
        extrinsic = extrinsic.reshape(-1, 4, 4)
        intrinsic = intrinsic.reshape(-1, 4, 4)
        gray_images = F.conv2d(src_images_tile, self.rgb2gray_filter)  # (SB, 1, H, W)
        pred_pc_world = self.pc_generator(gray_images, init_pclouds, extrinsic, intrinsic)  # (SB, N, 3)
        if self.freeze_graphx:
            pred_pc_world = pred_pc_world.detach()

        # rotate pclouds
        pred_pc_world = pred_pc_world.permute(0, 2, 1)  # (SB, N, 3) -> (SB, 3, N)
        rot_pred_pc_world = self.pc_rot_about_z[None] @ pred_pc_world

        # pclouds features
        self.pred_pc_world = rot_pred_pc_world  # (SB, 3, N)
        self.pc_features = self.pc_encoder(self.pred_pc_world)  # (SB, C, N)

        ret_dict = {
            'pc_world': pred_pc_world,  # (SB, 3, N)
            'voxel': gen_voxels  # torch.Size([batch_size, n_views, 32, 32, 32])
        }
        return ret_dict


    def forward(self, xyz_world, coarse=True, viewdirs=None):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz_world.shape
            NS = self.num_views_per_obj

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz_world, NS)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                # * Encode the xyz coordinates
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul(
                        self.poses[:, None, :3, :3], viewdirs
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat(
                        (z_feature, viewdirs), dim=1
                    )  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            # TODO: Order matters:
            ## xyz | vox_latent | pix_latent | pc_latent

            ##########################################
            # Voxel-aligned
            ##########################################
            if self.use_voxel_encoder:
                # xyz_world is in world, rotate it to align with GT mesh(voxel)
                xyz_rot = self.voxel_R_inv @ xyz_world.transpose(1, 2)  # (SB, 3, B)
                xyz_vox = xyz_rot.transpose(1, 2)  # (SB, B, 3)

                xyz_vox = xyz_vox[:, None, None, :, :]  # (SB, 1, 1, B, 3)

                if not self.multi_scale_query:
                    # directly query feature from voxel feature grid
                    latent = F.grid_sample(self.voxel_features, xyz_vox, padding_mode='zeros',
                                           align_corners=True)  # (SB, C, 1, 1, B)
                    latent = latent.transpose(1, 4).reshape(-1, self.d_vox_latent)  # (SB * B, C)
                else:
                    # From Geo-PIFu (multi-scale feature query)
                    # -----------------------------------------------------
                    # add displacements
                    xyz_vox = torch.cat([xyz_vox + d for d in self.displacments], dim=2)  # (B,1,7,B,3)

                    # grid sampling
                    latent = torch.nn.functional.grid_sample(self.voxel_features, xyz_vox, padding_mode='zeros',
                                                             align_corners=True)  # (SB,C=8,1,7,B)
                    latent = latent.transpose(1, 4).reshape(SB, B, -1)  # (SB, B, (7*C)=56)
                    latent = latent.reshape(-1, self.d_vox_latent)  # (SB * B, (7*C)=56)

                    # -----------------------------------------------------

                if self.freeze_voxel_encoder:
                    latent = latent.detach()

                mlp_input = torch.cat((mlp_input, latent), dim=-1)

            ##########################################
            # Pixel-aligned
            ##########################################
            if self.use_encoder:
                NS = self.num_views_per_obj

                # Grab encoder's latent code.
                uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
                uv *= repeat_interleave(
                    self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
                )
                uv += repeat_interleave(
                    self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
                )  # (SB * NS=1, B, 2)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB * NS=1, latent, B)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(
                    -1, self.d_pix_latent
                )  # (SB * NS=1 * B, latent)

                mlp_input = torch.cat((mlp_input, latent), dim=-1)

            ##########################################
            # pcloud-aligned
            ##########################################
            # add pc-aligned features
            # with torch.no_grad():  # no_grad (0513)
            #     dist = torch.cdist(xyz, self.src_pclouds.permute(0, 2, 1))
            # xyz_world is in world space, so is pred_pc_world
            dist = torch.cdist(xyz_world, self.pred_pc_world.permute(0, 2, 1))  # (0325)
            latent = self.agg_feats(dist, self.pc_features)
            latent = latent.reshape(-1, self.d_pc_latent)  # C is 128, (SB * B, C)

            if self.freeze_pc_encoder:
                latent = latent.detach()

            mlp_input = torch.cat((mlp_input, latent), dim=-1)

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    reshape_inner_dim=B
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    reshape_inner_dim=B
                )

            # Interpret the output
            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), self.sigma_act(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)
        return output
