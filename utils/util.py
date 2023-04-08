import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import math
import warnings
import yaml
from yacs.config import CfgNode as CN


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('INFO: initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def logistic_distrb(x):
    return torch.exp(x) / torch.pow(1 + torch.exp(x), 2)


def pc_normalize(pc):
    # pc shape: (B, 3, N)
    centroid = torch.mean(pc, dim=-1, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)), dim=-1).values  # m is (B,)
    pc = pc / m[:, None, None]

    return pc


def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    # from nerf_pl:
    # https://github.com/kwea123/nerf_pl/blob/b7e44f661b9c391cf10760bf19b9243c019cfbe8/utils/visualization.py#L6
    # img = np.nan_to_num(img)

    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


def cmap(img, color_map=cv2.COLORMAP_HOT):
    """
    Apply 'HOT' color to a float image
    """
    return cv2.applyColorMap(image_float_to_uint8(img), color_map)


def batched_index_select_nd(t, inds):
    """
    Index select on dim 1 of a n-dimensional batched tensor.
    :param t (batch, n, ...)
    :param inds (batch, k)
    :return (batch, k, ...)
    """
    return t.gather(
        1, inds[(...,) + (None,) * (len(t.shape) - 2)].expand(-1, -1, *t.shape[2:])
    )


def batched_index_select_nd_last(t, inds):
    """
    Index select on dim -1 of a >=2D multi-batched tensor. inds assumed
    to have all batch dimensions except one dataset dimension 'n'
    :param t (batch..., n, m)
    :param inds (batch..., k)
    :return (batch..., n, k)
    """
    dummy = inds.unsqueeze(-2).expand(*inds.shape[:-1], t.size(-2), inds.size(-1))
    out = t.gather(-1, dummy)
    return out


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
    )
    return transforms.Compose(ops)


def get_image_to_tensor(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), ]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )


def homogeneous(points):
    """
    Concat 1 to each point
    :param points (..., 3)
    :return (..., 4)
    """
    return F.pad(points, (0, 1), "constant", 1.0)


def gen_grid(*args, ij_indexing=False):
    """
    Generete len(args)-dimensional grid.
    Each arg should be (lo, hi, sz) so that in that dimension points
    are taken at linspace(lo, hi, sz).
    Example: gen_grid((0,1,10), (-1,1,20))
    :return (prod_i args_i[2], len(args)), len(args)-dimensional grid points
    """
    return torch.from_numpy(
        np.vstack(
            np.meshgrid(
                *(np.linspace(lo, hi, sz, dtype=np.float32) for lo, hi, sz in args),
                indexing="ij" if ij_indexing else "xy"
            )
        )
            .reshape(len(args), -1)
            .T
    )


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def coord_from_blender(dtype=torch.float32, device="cpu"):
    """
    Blender to standard coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def coord_to_blender(dtype=torch.float32, device="cpu"):
    """
    Standard to Blender coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def look_at(origin, target, world_up=np.array([0, 1, 0], dtype=np.float32)):
    """
    Get 4x4 camera to world space matrix, for camera looking at target
    """
    back = origin - target
    back /= np.linalg.norm(back)
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    cam_to_world = np.empty((4, 4), dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = back
    cam_to_world[:3, 3] = origin
    cam_to_world[3, :] = [0, 0, 0, 1]
    return cam_to_world


def get_cuda(gpu_id):
    """
    Get a torch.device for GPU gpu_id. If GPU not available,
    returns CPU device.
    """
    return (
        torch.device("cuda:%d" % gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def masked_sample(masks, num_pix, prop_inside, thresh=0.5):
    """
    :return (num_pix, 3)
    """
    num_inside = int(num_pix * prop_inside + 0.5)
    num_outside = num_pix - num_inside
    inside = (masks >= thresh).nonzero(as_tuple=False)
    outside = (masks < thresh).nonzero(as_tuple=False)

    pix_inside = inside[torch.randint(0, inside.shape[0], (num_inside,))]
    pix_outside = outside[torch.randint(0, outside.shape[0], (num_outside,))]
    pix = torch.cat((pix_inside, pix_outside))
    return pix


def bbox_sample(bboxes, num_pix):
    """
    :return (num_pix, 3)
    """
    image_ids = torch.randint(0, bboxes.shape[0], (num_pix,), device=bboxes.device)
    pix_bboxes = bboxes[image_ids]
    x = (
            torch.rand(num_pix, device=bboxes.device) * (pix_bboxes[:, 2] + 1 - pix_bboxes[:, 0])
            + pix_bboxes[:, 0]
    ).long()
    y = (
            torch.rand(num_pix, device=bboxes.device) * (pix_bboxes[:, 3] + 1 - pix_bboxes[:, 1])
            + pix_bboxes[:, 1]
    ).long()
    pix = torch.stack((image_ids, y, x), dim=-1)
    return pix


def ndc_rays(W, H, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
            .unsqueeze(0)
            .repeat(num_images, 1, 1, 1)
    )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]
    if ndc:
        if not (z_near == 0 and z_far == 1):
            warnings.warn(
                "dataset z near and z_far not compatible with NDC, setting them to 0, 1 NOW"
            )
        z_near, z_far = 0.0, 1.0
        cam_centers, cam_raydir = ndc_rays(
            width, height, focal, 1.0, cam_centers, cam_raydir
        )

    cam_nears = (
        torch.tensor(z_near, device=device)
            .view(1, 1, 1, 1)
            .expand(num_images, height, width, -1)
    )
    cam_fars = (
        torch.tensor(z_far, device=device)
            .view(1, 1, 1, 1)
            .expand(num_images, height, width, -1)
    )
    return torch.cat(
        (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
    )  # (B, H, W, 8)


def trans_t(t):
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1], ], dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
            torch.tensor(
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            @ c2w
    )
    return c2w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def make_conv_2d(
        dim_in,
        dim_out,
        padding_type="reflect",
        norm_layer=None,
        activation=None,
        kernel_size=3,
        use_bias=False,
        stride=1,
        no_pad=False,
        zero_init=False,
):
    conv_block = []
    amt = kernel_size // 2
    if stride > 1 and not no_pad:
        raise NotImplementedError(
            "Padding with stride > 1 not supported, use same_pad_conv2d"
        )

    if amt > 0 and not no_pad:
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(amt)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(amt)]
        elif padding_type == "zero":
            conv_block += [nn.ZeroPad2d(amt)]
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

    conv_block.append(
        nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, bias=use_bias, stride=stride
        )
    )
    if zero_init:
        nn.init.zeros_(conv_block[-1].weight)
    #  else:
    #  nn.init.kaiming_normal_(conv_block[-1].weight)
    if norm_layer is not None:
        conv_block.append(norm_layer(dim_out))

    if activation is not None:
        conv_block.append(activation)
    return nn.Sequential(*conv_block)


def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:]
    out_height = math.ceil(in_height / stride)
    out_width = math.ceil(in_width / stride)

    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def same_pad_conv2d(t, padding_type="reflect", kernel_size=3, stride=1, layer=None):
    """
    Perform SAME padding on tensor, given kernel size/stride of conv operator
    assumes kernel/stride are equal in all dimensions.
    Use before conv called.
    Dilation not supported.
    :param t image tensor input (B, C, H, W)
    :param padding_type padding type constant | reflect | replicate | circular
    constant is 0-pad.
    :param kernel_size kernel size of conv
    :param stride stride of conv
    :param layer optionally, pass conv layer to automatically get kernel_size and stride
    (overrides these)
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    return F.pad(
        t, calc_same_pad_conv2d(t.shape, kernel_size, stride), mode=padding_type
    )


def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    """
    Perform SAME unpad on tensor, given kernel/stride of deconv operator.
    Use after deconv called.
    Dilation not supported.
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d(
        (h_scaled, w_scaled), kernel_size, stride
    )
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def psnr(pred, target):
    """
    Compute PSNR of two tensors in decibels.
    pred/target should be of same size or broadcastable
    """
    return -10 * torch.log10(torch.mean((pred - target) ** 2))


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


def rot_to_quat(R):
    """
    Rotation matrix to quaternion
    """
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


###################################################################################################
# Handy functions
###################################################################################################
from functools import wraps
import time
import os
import os.path as osp
from datetime import datetime
import imageio


def timeit(func):
    @wraps(func)
    def timed(*args, **kw):
        tstart = time.time()
        output = func(*args, **kw)
        tend = time.time()

        print('INFO: "{}" took {:.3f} s to execute\n'.format(func.__name__, (tend - tstart)))
        return output

    return timed


###################################################################################################
def getattr_tree(cfg, name):
    return functools.reduce(lambda cfg, name: getattr(cfg, name), [cfg] + name.split('.'))


def compile_remarks(cfg):
    remarks = ''
    for k, v in cfg.remarks_dict.items():
        if k != v.split('.')[-1]:
            remarks += f'_{k}{getattr_tree(cfg, v)}'
        else:
            param = getattr_tree(cfg, v)
            remarks += f'_{param}' if param is not None else ''

    ori_remarks = cfg.remarks
    if ori_remarks:
        final_remarks = ori_remarks + remarks
    else:
        final_remarks = remarks[1:]  # remove the first underline

    cfg.remarks = final_remarks


def create_logdir(c=None, resume_path=None, args=None):
    if not resume_path:
        model_name = c.trainer.model
        exp_name = f'{model_name}_{c.remarks}' if c.remarks else model_name
        exp_name = f'Debug_{exp_name}' if args.debug else exp_name

        folder_name = f'{exp_name}_{datetime.now().strftime(c.trainer.time_format)}'
        folder_name = f'Dryrun_{folder_name}' if c.trainer.fast_dev_run else folder_name

        work_dir = osp.join(c.trainer.base_dir, folder_name)
        vis_dir = osp.join(work_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        print(f'INFO: create log dir: {work_dir}')

        c.exp_name = exp_name
        c.work_dir = work_dir
        c.vis_dir = vis_dir
    else:
        work_dir = osp.dirname(osp.dirname(resume_path))
        cfg_path = osp.join(work_dir, 'config.yaml')
        c = load_config(cfg_path)

        return c


def store_cmd(c):
    c.command_line = ' '.join(['python'] + sys.argv)


def load_config(cfg_path):
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)

    _C = CN(new_allowed=True)
    if cfg_dict['inherit_from'] is not None:
        _C_inherit_from = load_config(cfg_dict['inherit_from'])
        _C.merge_from_other_cfg(_C_inherit_from)
        _C.merge_from_file(cfg_path)
    else:
        _C.merge_from_file(cfg_path)

    return _C


def save_config(cfg):
    save_path = osp.join(cfg.work_dir, 'config.yaml')
    with open(save_path, 'w') as yaml_file:
        cfg.dump(stream=yaml_file, indent=4)
    print(f'INFO: config yaml saved to {save_path}')


def print_config(cfg):
    message = ''
    message += '----------------- Configs ---------------\n'
    for k, v in sorted(vars(cfg).items()):
        if k == 'remarks_dict':
            continue
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)


def flat_config(obj, keep_lists=True, max_split_size=10, flatten_int=False):
    def items():
        for key, val in obj.items():
            if isinstance(val, dict) and (isinstance(key, str) or (isinstance(key, int) and flatten_int)):
                intermediate_dict = {}
                for subkey, subval in flat_config(val).items():
                    if isinstance(subkey, str):
                        yield str(key) + "." + subkey, subval
                    elif isinstance(subkey, int) and flatten_int:
                        yield str(key) + "." + str(subkey), subval
                    else:
                        intermediate_dict[subkey] = subval
                if len(intermediate_dict) > 0:
                    yield str(key), intermediate_dict
            elif isinstance(val, (list, tuple)):
                keep_this = (
                        keep_lists or not isinstance(key, (str, int)) or (isinstance(key, int) and not flatten_int)
                )
                if max_split_size not in (None, False) and len(val) > max_split_size:
                    keep_this = True
                if keep_this:
                    yield key, val
                else:
                    for i, subval in enumerate(val):
                        yield str(key) + "." + str(i), subval
            else:
                yield key, val

    return dict(items())


def merge_from_sys_argv(config, warn=False):
    """Updates Config with the arguments passed as args when running the
    program. Keys will be converted to command line options, then matching
    options in `sys.argv` will be used to update the Config.

    Args:
        config (Config): Update this Config.
        warn (bool): Raise warnings if there are unknown options. Turn this on
            if you don't use any :class:`argparse.ArgumentParser` after to
            check for possible errors.

    """

    import sys
    import argparse
    import warnings

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser(allow_abbrev=False)

        config_flat = flat_config(config)
        for key, val in config_flat.items():
            name = "--{}".format(key)
            if val is None:
                parser.add_argument(name)
            else:
                if type(val) == bool:
                    parser.add_argument(name, type=str2bool, default=val)
                elif isinstance(val, (list, tuple)):
                    if len(val) > 0 and type(val[0]) != type:
                        parser.add_argument(name, nargs="+", type=type(val[0]), default=val)
                    else:
                        parser.add_argument(name, nargs="+", default=val)
                else:
                    parser.add_argument(name, type=type(val), default=val)

        # parse args
        param, unknown = parser.parse_known_args()
        param = vars(param)

        if len(unknown) > 0 and warn:
            warnings.warn("Called with unknown arguments: {}".format(unknown), RuntimeWarning)

        param_list = functools.reduce(lambda x, y: x + y, param.items())
        config.merge_from_list(param_list)


def save_img(path, img):
    imageio.imwrite(path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
