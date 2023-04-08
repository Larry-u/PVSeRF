import os
import os.path as osp
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from utils import get_image_to_tensor_balanced, get_mask_to_tensor


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


class SRN(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(self, dset_cfg, phase="train"):
        """
        :param phase train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.dset_cfg = dset_cfg
        phase = 'val_subset' if phase == 'val' else phase
        self.base_path = dset_cfg.base_path.format(dset_cfg.cat_name, dset_cfg.cat_name, phase)
        assert osp.exists(self.base_path)

        self.dataset_name = os.path.basename(self.base_path)
        self.phase = phase

        is_chair = "chair" in self.dataset_name
        if is_chair and phase == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        # TODO: comment the following!
        # with open('./finish.txt', 'r') as f:
        #     finish_objs = [x.strip() for x in f.readlines()]
        #     print(f'INFO: {len(finish_objs)} objs will be skipped...')
        # self.intrins = [x for x in self.intrins if osp.basename(osp.dirname(x)) not in finish_objs]
        # TODO: comment the above

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = dset_cfg.image_size
        self.world_scale = dset_cfg.world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = self.dset_cfg.z_near
        self.z_far = self.dset_cfg.z_far
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans

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

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result


class SRNGraphxWorld(SRN):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(self, dset_cfg, phase='train'):
        super(SRNGraphxWorld, self).__init__(dset_cfg, phase)
        self.N_points = dset_cfg.N_points

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        dvr_focal = focal / width * 2
        intrinsic = torch.tensor([[dvr_focal, 0, 0, 0],
                                  [0, dvr_focal, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], dtype=torch.float32)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        all_intrinsics = [intrinsic] * len(rgb_paths)
        all_extrinsics = []

        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose_np = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            pose = torch.from_numpy(pose_np)
            pose = pose @ self._coord_trans

            extrinsic = np.linalg.inv(pose_np)
            all_extrinsics.append(torch.from_numpy(extrinsic))

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

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)
        all_extrinsics = torch.stack(all_extrinsics)
        all_intrinsics = torch.stack(all_intrinsics)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        # prepare init pointcloud for GraphX
        point_cloud = torch.from_numpy(init_pointcloud_loader_NMR_world(self.N_points))

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
            'point_cloud': point_cloud,
            'extrinsics': all_extrinsics,
            'intrinsics': all_intrinsics
        }
        return result
