import os
import cv2
import os.path as osp
from glob import glob
from random import shuffle
from concurrent import futures

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import imageio
import numpy as np

from utils import get_image_to_tensor_balanced, get_mask_to_tensor, get_image_to_tensor

CENTER = 31.5
FOCAL = 119.42


def init_pointcloud_loader_NMR(num_points):
    # Z = np.random.rand(num_points) + 1.
    Z = np.random.uniform(2., 4., size=(num_points,))
    h = np.random.uniform(2., 62., size=(num_points,))
    w = np.random.uniform(2., 62., size=(num_points,))
    X = (w - CENTER) / FOCAL * -Z
    Y = (h - CENTER) / FOCAL * Z
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')


def init_pointcloud_loader_NMR_world(num_points):
    # Z = np.random.rand(num_points) + 1.
    X = np.random.uniform(-0.8, 0.8, size=(num_points,))
    Y = np.random.uniform(-0.8, 0.8, size=(num_points,))
    Z = np.random.uniform(-0.8, 0.8, size=(num_points,))

    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')


class NMR(Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(self, dset_cfg, phase='train'):
        super().__init__()
        self.dset_cfg = dset_cfg
        self.base_path = dset_cfg.base_path
        assert osp.exists(self.base_path)

        cats = [x for x in glob(osp.join(self.base_path, "*")) if osp.isdir(x)]

        if phase == "train":
            file_lists = [osp.join(x, dset_cfg.list_prefix + "train.lst") for x in cats]
        elif phase == "val":
            file_lists = [osp.join(x, dset_cfg.list_prefix + "val_subset.lst") for x in cats]
            # file_lists = [osp.join(x, dset_cfg.list_prefix + "val.lst") for x in cats]
        elif phase == "test":
            file_lists = [osp.join(x, dset_cfg.list_prefix + "test.lst") for x in cats]

        all_objs = []
        for file_list in file_lists:
            if not osp.exists(file_list):
                print(f'WARNING: {file_list} not exist')
                continue
            base_dir = osp.dirname(file_list)
            cat = osp.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, osp.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        # shuffle all_objs in val phase to avoid validating on same category
        if phase == 'val': shuffle(all_objs)

        # shuffle(all_objs)  # only shuffle when want to test model on trainset

        self.all_objs = all_objs
        self.phase = phase

        self.image_to_tensor = get_image_to_tensor_balanced()  # [-1, 1]
        # self.image_to_tensor = get_image_to_tensor()  # [0, 1]
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = dset_cfg.image_size
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self.scale_focal = dset_cfg.scale_focal
        self.max_imgs = dset_cfg.max_imgs

        self.z_near = dset_cfg.z_near
        self.z_far = dset_cfg.z_far
        self.lindisp = False

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob(osp.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob(osp.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        assert len(rgb_paths) and len(mask_paths)

        cam_path = osp.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            # ShapeNet
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

            intr_mtx = all_cam["camera_mat_" + str(i)]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - fy) < 1e-9
            fx = fx * x_scale
            if focal is None:
                focal = fx
            else:
                assert abs(fx - focal) < 1e-5
            pose = extr_inv_mtx

            pose = (
                    self._coord_trans_world
                    @ torch.tensor(pose, dtype=torch.float32)
                    @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if len(all_bboxes) > 0:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if len(all_bboxes) > 0:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "camera_mat": torch.tensor(intr_mtx, dtype=torch.float32)
        }
        if all_masks is not None:
            result["masks"] = all_masks

        result["bbox"] = all_bboxes

        return result


class NMRDepth(NMR):
    """
    Inherit from NMR, but getitem will return depth map and intrinsic and extrinsic matrices
    """

    def get_disparity(self, depth, use_disparity=False, norm=False, replace_inf=0):
        disparity = depth.copy()
        if use_disparity:
            disparity[disparity != np.inf] = 1 / disparity[disparity != np.inf]

        if norm:
            valid_disparity = disparity[disparity != np.inf]
            mi, ma = valid_disparity.min(), valid_disparity.max()
            # norm to 0~1
            disparity[disparity != np.inf] = (disparity[disparity != np.inf] - mi) / (ma - mi + 1e-8)

        if replace_inf is not None:
            # replace inf with 0 by default
            disparity[disparity == np.inf] = replace_inf

        return disparity

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob(osp.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob(osp.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        depth_folder_name = 'visual_hull_depth'
        depth_paths = sorted(glob(osp.join(root_dir, f'{depth_folder_name}/*.exr')))

        # in DTU dataset, #images in each scan are different,
        # so sample same number of imgs to composite a batch
        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]
            depth_paths = [depth_paths[i] for i in sel_indices]

        assert len(rgb_paths) and len(mask_paths) and len(depth_paths)

        cam_path = osp.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        all_intrin_inv = []
        all_extrin_inv = []
        all_depths = []
        all_disparitys = []

        focal = None

        for idx, (rgb_path, mask_path, depth_path) in enumerate(zip(rgb_paths, mask_paths, depth_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]

            # read depth
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)
            depth = depth.reshape(depth.shape[0], depth.shape[1], -1)[:, :, 0]
            all_depths.append(torch.from_numpy(depth))

            # get disparity
            disparity = self.get_disparity(depth, use_disparity=True, norm=False)
            all_disparitys.append(torch.from_numpy(disparity))

            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            # ShapeNet
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

            intr_mtx = all_cam["camera_mat_" + str(i)]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - fy) < 1e-9
            fx = fx * x_scale
            if focal is None:
                focal = fx
            else:
                assert abs(fx - focal) < 1e-5
            pose = extr_inv_mtx

            intrinsic_inv = all_cam['camera_mat_inv_{}'.format(i)]
            extrinsic_inv = all_cam['world_mat_inv_{}'.format(i)]

            all_intrin_inv.append(torch.tensor(intrinsic_inv, dtype=torch.float32))
            all_extrin_inv.append(torch.tensor(extrinsic_inv, dtype=torch.float32))

            tl = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            tr = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            pose = (
                    self._coord_trans_world
                    @ torch.tensor(pose, dtype=torch.float32)
                    @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if len(all_bboxes) > 0:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_intrin_inv = torch.stack(all_intrin_inv)
        all_extrin_inv = torch.stack(all_extrin_inv)
        all_depths = torch.stack(all_depths)
        all_disparitys = torch.stack(all_disparitys)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if len(all_bboxes) > 0:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            'intrin_invs': all_intrin_inv,
            'extrin_invs': all_extrin_inv,
            'depths': all_depths,
            'disparitys': all_disparitys,
            "camera_mat": torch.tensor(intr_mtx, dtype=torch.float32)
        }
        if all_masks is not None:
            result["masks"] = all_masks

        result["bbox"] = all_bboxes

        return result


class NMRCloud(NMR):
    """
    Inherit from NMR, but getitem will return N_points GT-pcloud sampled from 10k points
    """

    def __init__(self, dset_cfg, phase='train'):
        super(NMRCloud, self).__init__(dset_cfg, phase)
        self.N_points = dset_cfg.N_points
        self.rot_pc = dset_cfg.rot_pc

        self.rot_about_z = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.rot_about_y = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob(osp.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob(osp.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        assert len(rgb_paths) and len(mask_paths)

        cam_path = osp.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            # ShapeNet
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

            intr_mtx = all_cam["camera_mat_" + str(i)]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - fy) < 1e-9
            fx = fx * x_scale
            if focal is None:
                focal = fx
            else:
                assert abs(fx - focal) < 1e-5
            pose = extr_inv_mtx

            pose = (
                    self._coord_trans_world
                    @ torch.tensor(pose, dtype=torch.float32)
                    @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if len(all_bboxes) > 0:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if len(all_bboxes) > 0:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        pc_path = osp.join(root_dir, 'pointcloud.npz')
        point_cloud = np.load(pc_path)['points']
        sel_inds = np.random.choice(point_cloud.shape[0], self.N_points, replace=False)  # FIXME: forget replace=False?
        point_cloud = point_cloud[sel_inds]  # (N,3)
        if self.rot_pc:
            point_cloud = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1))], axis=-1)  # (N,4)
            point_cloud = self.rot_about_y @ self.rot_about_z @ point_cloud.T  # (4, N)
            point_cloud = point_cloud[:3].T  # (N,3)

        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            'point_cloud': point_cloud
        }
        if all_masks is not None:
            result["masks"] = all_masks

        result["bbox"] = all_bboxes

        return result


class NMRGraphX(NMR):
    """
    Inherit from NMR, but getitem will return N_points GT-pcloud sampled from 10k points
    """

    def __init__(self, dset_cfg, phase='train'):
        super(NMRGraphX, self).__init__(dset_cfg, phase)
        self.N_points = dset_cfg.N_points

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob(osp.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob(osp.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        assert len(rgb_paths) and len(mask_paths)

        cam_path = osp.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            # ShapeNet
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

            intr_mtx = all_cam["camera_mat_" + str(i)]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - fy) < 1e-9
            fx = fx * x_scale
            if focal is None:
                focal = fx
            else:
                assert abs(fx - focal) < 1e-5
            pose = extr_inv_mtx

            pose = (
                    self._coord_trans_world
                    @ torch.tensor(pose, dtype=torch.float32)
                    @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if len(all_bboxes) > 0:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if len(all_bboxes) > 0:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        # prepare init pointcloud for GraphX
        point_cloud = torch.from_numpy(init_pointcloud_loader_NMR(self.N_points))

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            'point_cloud': point_cloud
        }
        if all_masks is not None:
            result["masks"] = all_masks

        result["bbox"] = all_bboxes

        return result


class NMRGraphXWorld(NMR):
    """
    Inherit from NMR, but getitem will return N_points GT-pcloud sampled from 10k points
    """

    def __init__(self, dset_cfg, phase='train'):
        super(NMRGraphXWorld, self).__init__(dset_cfg, phase)
        self.N_points = dset_cfg.N_points

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob(osp.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob(osp.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        assert len(rgb_paths) and len(mask_paths)

        cam_path = osp.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        all_extrinsics = []
        all_intrinsics = []
        focal = None

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            # ShapeNet
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

            intr_mtx = all_cam["camera_mat_" + str(i)]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - fy) < 1e-9
            fx = fx * x_scale
            if focal is None:
                focal = fx
            else:
                assert abs(fx - focal) < 1e-5
            pose = extr_inv_mtx

            pose = (
                    self._coord_trans_world
                    @ torch.tensor(pose, dtype=torch.float32)
                    @ self._coord_trans_cam
            )

            extrinsic = torch.tensor(all_cam[f'world_mat_{i}'].astype(np.float32))
            intrinsic = torch.tensor(intr_mtx.astype(np.float32))

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)
            all_extrinsics.append(extrinsic)
            all_intrinsics.append(intrinsic)

        if len(all_bboxes) > 0:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_extrinsics = torch.stack(all_extrinsics)
        all_intrinsics = torch.stack(all_intrinsics)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if len(all_bboxes) > 0:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        # prepare init pointcloud for GraphX
        point_cloud = torch.from_numpy(init_pointcloud_loader_NMR_world(self.N_points))

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            'point_cloud': point_cloud,
            'extrinsics': all_extrinsics,
            'intrinsics': all_intrinsics
        }
        if all_masks is not None:
            result["masks"] = all_masks

        result["bbox"] = all_bboxes

        return result


