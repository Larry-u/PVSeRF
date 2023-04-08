import os

import torch
import numpy as np
from dotmap import DotMap

from pl_models import PCloudNeRF
import utils
from utils.visualize import visualize_data, visualize_pc_and_voxel


class PVSeRF(PCloudNeRF):
    '''
    Use voxel-aligned features
    voxel is predicted from single-view image (or along with depth map)

    '''

    # def training_step(self, batch, batch_idx, optimizer_idx=None):
    def training_step(self, batch, batch_idx):
        if "images" not in batch:
            return {}
        all_images = batch["images"]  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = batch["poses"]  # (SB, NV, 4, 4)
        all_bboxes = batch.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = batch["focal"]  # (SB)
        all_c = batch.get("c")  # (SB)
        all_pc = batch['point_cloud']  # (SB, N_points, 3)
        all_extrinsics = batch['extrinsics']  # (SB, NV, 4, 4)
        all_intrinsics = batch['intrinsics']  # (SB, NV, 4, 4)

        if self.use_bbox and self.global_step >= self.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", self.global_step)

        if not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), ()).item()]

        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)

        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if "c" in batch:
                c = batch["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            cam_rays = utils.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            )  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = utils.bbox_sample(bboxes, self.dset_cfg.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (self.dset_cfg.ray_batch_size,))

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=self.device
            )  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(self.device)
        src_images = utils.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = utils.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)
        src_extrin = utils.batched_index_select_nd(all_extrinsics, image_ord)  # (SB, NS, 4, 4)
        src_intrin = utils.batched_index_select_nd(all_intrinsics, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        self.nerf_net.encode(
            src_images,
            src_poses,
            all_pc,
            all_focals.to(device=self.device),
            c=all_c.to(device=self.device) if all_c is not None else None,
            extrinsic=src_extrin,
            intrinsic=src_intrin
        )

        render_dict = DotMap(self(all_rays, want_weights=True))
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        self.log('rc', rgb_loss.item() * self.lambda_coarse)
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            self.log('rf', fine_loss.item() * self.lambda_fine)

        loss = rgb_loss
        self.log('t', loss.item())

        return loss

    # first method: save each val vis in validation_step (but batch_idx will overlap when using two gpus with ddp)
    def validation_step(self, batch, batch_idx):
        if "images" not in batch:
            return {}
        images = batch["images"][0]  # (NV, 3, H, W)
        poses = batch["poses"][0]  # (NV, 4, 4)
        focal = batch["focal"][0]  # (1)
        pclouds = batch['point_cloud']  # (1, N_points, 3)
        all_extrinsics = batch['extrinsics'][0]  # (NV, 4, 4)
        all_intrinsics = batch['intrinsics'][0]  # (NV, 4, 4)

        c = batch.get("c")
        # if c is not None:
        #     c = c[0]  # (1)
        NV, _, H, W = images.shape
        cam_rays = utils.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=None if c is None else c[0]
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        source_views = (
            images_0to1[views_src]
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).reshape(H, W, 3)

        test_rays = cam_rays[view_dest]  # (H, W, 8)
        test_images = images[views_src]  # (NS, 3, H, W)

        self.nerf_net.encode(
            test_images.unsqueeze(0),
            poses[views_src].unsqueeze(0),
            pclouds,
            focal.to(self.device),
            c=c.to(self.device) if c is not None else None,
            extrinsic=all_extrinsics[views_src].unsqueeze(0),
            intrinsic=all_intrinsics[views_src].unsqueeze(0)
        )
        test_rays = test_rays.reshape(1, H * W, -1)
        render_dict = DotMap(self(test_rays, want_weights=True))
        coarse = render_dict.coarse
        fine = render_dict.fine

        using_fine = len(fine) > 0

        alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
        depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)
        rgb_coarse = coarse.rgb[0].reshape(H, W, 3)

        # print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        # print("c alpha min {}, max {}".format(alpha_coarse_np.min(), alpha_coarse_np.max()))

        alpha_coarse_cmap = utils.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = utils.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt.cpu().numpy(),
            depth_coarse_cmap,
            rgb_coarse.cpu().numpy(),
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
            depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
            rgb_fine = fine.rgb[0].reshape(H, W, 3)
            # print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            # print("f alpha min {}, max {}".format(alpha_fine_np.min(), alpha_fine_np.max()))

            alpha_fine_cmap = utils.cmap(alpha_fine_np) / 255
            depth_fine_cmap = utils.cmap(depth_fine_np) / 255
            vis_list = [
                *source_views,
                gt.cpu().numpy(),
                depth_fine_cmap,
                rgb_fine.cpu().numpy(),
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine
        else:
            rgb_psnr = rgb_coarse

        psnr = utils.psnr(rgb_psnr, gt)
        self.log('cur_psnr', psnr, on_epoch=False, on_step=True, prog_bar=True, logger=False)

        dpath = batch["path"][0]
        obj_basename = os.path.basename(dpath)
        cat_name = os.path.basename(os.path.dirname(dpath))
        obj_name = cat_name + "_" + obj_basename

        return {"vis": vis, "psnr": psnr, "obj_name": obj_name}

    def test_init(self, args):
        super(PVSeRF, self).test_init(args)
        self.vis_geometry = vars(args).get('vis_geometry')

    def test_step(self, batch, batch_idx):
        dpath = batch["path"][0]
        obj_basename = os.path.basename(dpath)
        cat_name = os.path.basename(os.path.dirname(dpath))
        obj_name = cat_name + "_" + obj_basename if self.multicat else obj_basename
        images = batch["images"][0]  # (NV, 3, H, W), [-1, 1]

        NV, _, H, W = images.shape

        if self.use_source_lut or self.free_pose:
            if self.use_source_lut:
                obj_id = cat_name + "/" + obj_basename
                self.source = self.source_lut[obj_id]

        NS = len(self.source)
        src_view_mask = torch.zeros(NV, dtype=torch.bool)
        src_view_mask[self.source] = 1

        focal = batch["focal"][0]
        if isinstance(focal, float):
            focal = torch.tensor(focal, dtype=torch.float32)
        focal = focal[None]

        c = batch.get("c")
        # if c is not None:
        #     c = c[0].to(self.device).unsqueeze(0)  #FIXME: c[0].unsqueeze(0) == c ??? (if bs=1)

        poses = batch["poses"][0]  # (NV, 4, 4)
        src_poses = poses[src_view_mask]  # (NS, 4, 4)

        target_view_mask = torch.ones(NV, dtype=torch.bool)
        if not self.include_src:
            target_view_mask *= ~src_view_mask

        novel_view_idxs = target_view_mask.nonzero(as_tuple=False).reshape(-1)

        poses = poses[target_view_mask]  # (NV[-NS], 4, 4)

        all_rays = utils.gen_rays(
            poses.reshape(-1, 4, 4), W, H, focal, self.z_near, self.z_far, c=c
        ).reshape(-1, 8)  # ((NV[-NS])*H*W, 8)

        poses = None

        n_gen_views = len(novel_view_idxs)

        pclouds = batch['point_cloud']  # (1, N_points, 3)
        all_extrinsics = batch['extrinsics'][0]  # (NV, 4, 4)
        all_intrinsics = batch['intrinsics'][0]  # (NV, 4, 4)
        ret_dict = self.nerf_net.encode(
            images[src_view_mask].unsqueeze(0),
            src_poses.unsqueeze(0),
            pclouds,
            focal.to(device=self.device),
            c=c.to(device=self.device) if c is not None else None,
            extrinsic=all_extrinsics[src_view_mask].unsqueeze(0),
            intrinsic=all_intrinsics[src_view_mask].unsqueeze(0)
        )

        # prepare outdir
        obj_out_dir = os.path.join(self.test_results_dir, obj_name)
        os.makedirs(obj_out_dir, exist_ok=True)

        # visualize pc/voxel
        if self.vis_geometry:
            pred_pc_world = ret_dict['pc_world'][0].transpose(0, 1).cpu().numpy()  # (N, 3)
            gen_voxel = ret_dict['voxel'][0][0].cpu().numpy()  # (32, 32, 32)
            gen_voxel = gen_voxel.squeeze().__ge__(0.5)

            visualize_data(pred_pc_world, 'pointcloud', os.path.join(obj_out_dir, 'pointcloud.png'))
            visualize_data(gen_voxel, 'voxels', os.path.join(obj_out_dir, 'voxel.png'))
            visualize_pc_and_voxel((pred_pc_world + 0.5) * 32, gen_voxel, out_file=os.path.join(obj_out_dir, 'pc_voxel.png'))
            return {"ssim": 0, "psnr": 0}  # FIXME: temp code to make test_step fast, remove it in normal testing

        if all_rays.size(0) > self.eval_ray_batch_size:
            rays_spl = torch.split(all_rays, self.eval_ray_batch_size, dim=0)  # Creates views

            all_rgb, all_depth = [], []
            for rays in rays_spl:
                rgb, depth = self(rays[None])
                rgb = rgb[0]
                depth = depth[0]
                all_rgb.append(rgb)
                all_depth.append(depth)

            all_rgb = torch.cat(all_rgb, dim=0)
            all_depth = torch.cat(all_depth, dim=0)
        else:
            all_rgb, all_depth = self(all_rays[None])
            all_rgb, all_depth = all_rgb[0], all_depth[0]

        all_depth = (all_depth - self.z_near) / (self.z_far - self.z_near)
        all_depth = all_depth.reshape(n_gen_views, H, W).cpu().numpy()

        all_rgb = torch.clamp(
            all_rgb.reshape(n_gen_views, H, W, 3), 0.0, 1.0
        ).cpu().numpy()  # (NV-NS, H, W, 3)

        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)
        images_gt = images_0to1[target_view_mask]
        rgb_gt_all = (
            images_gt.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )  # (NV-NS, H, W, 3)

        # write prediction images
        for i in range(n_gen_views):
            out_file = os.path.join(
                obj_out_dir, "{:06}.png".format(novel_view_idxs[i].item())
            )
            utils.save_img(out_file, (all_rgb[i] * 255).astype(np.uint8))

            if self.write_depth:
                out_depth_file = os.path.join(
                    obj_out_dir, "{:06}_depth.exr".format(novel_view_idxs[i].item())
                )
                out_depth_norm_file = os.path.join(
                    obj_out_dir,
                    "{:06}_depth_norm.png".format(novel_view_idxs[i].item()),
                )
                depth_cmap_norm = utils.cmap(all_depth[i])
                utils.save_exr(out_depth_file, all_depth[i])
                utils.save_img(out_depth_norm_file, depth_cmap_norm)

            if self.write_compare:
                out_file = os.path.join(
                    obj_out_dir,
                    "{:06}_compare.png".format(novel_view_idxs[i].item()),
                )
                depth_cmap_norm = utils.cmap(all_depth[i]) / 255
                out_im = np.hstack((depth_cmap_norm, all_rgb[i], rgb_gt_all[i]))
                utils.save_img(out_file, (out_im * 255).astype(np.uint8))

        curr_ssim = 0.0
        curr_psnr = 0.0
        if not self.no_compare_gt:
            for view_idx in range(n_gen_views):
                ssim = utils.compare_ssim(
                    all_rgb[view_idx],
                    rgb_gt_all[view_idx],
                    multichannel=True,
                    data_range=1,
                )
                psnr = utils.compare_psnr(
                    all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                )
                curr_ssim += ssim
                curr_psnr += psnr

                if self.write_compare:
                    out_file = os.path.join(
                        obj_out_dir,
                        "{:06}_compare.png".format(novel_view_idxs[view_idx].item()),
                    )
                    depth_cmap_norm = utils.cmap(all_depth[i]) / 255
                    out_im = np.hstack((depth_cmap_norm, all_rgb[i], rgb_gt_all[i]))
                    utils.save_img(out_file, (out_im * 255).astype(np.uint8))

        curr_psnr /= n_gen_views
        curr_ssim /= n_gen_views

        return {"ssim": curr_ssim, "psnr": curr_psnr}

PVSeRFSRN = PVSeRF