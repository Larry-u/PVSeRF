import os
import os.path as osp

import torch
from torch import optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap

import dataset
from networks import loss, nerf_nets
from render import NeRFRenderer
import utils


class PCloudNeRF(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dset_cfg = cfg.dataset
        self.train_cfg = cfg.trainer

        if self.cfg.renderer.ndc:
            cfg.model.norm_pc = True

        self.nerf_net = getattr(nerf_nets, f'{cfg.trainer.model}Net')(cfg.model)
        self.renderer = NeRFRenderer(cfg.renderer)

        self.nviews = list(map(int, cfg.trainer.nviews.split()))

        self.lambda_coarse = cfg.loss.lambda_coarse
        self.lambda_fine = cfg.loss.lambda_fine
        print(
            "INFO: lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )

        self.rgb_coarse_crit = loss.get_rgb_loss(cfg.loss.rgb, True)
        fine_loss_conf = cfg.loss.rgb
        if "rgb_fine" in cfg.loss:
            print("INFO: using fine loss")
            fine_loss_conf = cfg.loss.rgb_fine
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        self.z_near = self.dset_cfg.z_near
        self.z_far = self.dset_cfg.z_far

        self.use_bbox = self.train_cfg.no_bbox_step > 0
        self.no_bbox_step = self.train_cfg.no_bbox_step

        self.simple_output = False  # by default

    def configure_optimizers(self):
        return optim.Adam(self.nerf_net.parameters(), lr=self.train_cfg.lr)

    def on_train_epoch_start(self):
        print(f'\n\nexp = {self.cfg.work_dir}')

    def forward(self, rays, want_weights=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )

        outputs = self.renderer(
            self.nerf_net, rays, want_weights=want_weights and not self.simple_output
        )
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()

    # def training_step(self, batch, batch_idx, optimizer_idx=None):
    def training_step(self, batch, batch_idx):
        if "images" not in batch:
            return {}
        all_images = batch["images"]  # (SB, NV, 3, H, W), [-1, 1]
        all_depths = batch["depths"]  # (SB, NV, H, W)
        all_intrin_inv = batch["intrin_invs"]  # (SB, NV, 4, 4)
        all_extrin_inv = batch["extrin_invs"]  # (SB, NV, 4, 4)

        SB, NV, _, H, W = all_images.shape
        all_poses = batch["poses"]  # (SB, NV, 4, 4)
        all_bboxes = batch.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = batch["focal"]  # (SB)
        all_c = batch.get("c")  # (SB)

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
            images = all_images[obj_idx]  # (NV, 3, H, W), [-1, 1]
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
                poses, W, H, focal, self.z_near, self.z_far, c=c, ndc=self.cfg.renderer.ndc
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

        src_depths = utils.batched_index_select_nd(all_depths, image_ord)  # (SB, NS, H, W)
        src_extrin_invs = utils.batched_index_select_nd(all_extrin_inv, image_ord)  # (SB, NS, 4, 4)
        src_intrin_invs = utils.batched_index_select_nd(all_intrin_inv, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        self.nerf_net.encode(
            src_depths,
            src_extrin_invs,
            src_intrin_invs,
            src_images
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

    def validation_step(self, batch, batch_idx):
        if "images" not in batch:
            return {}
        images = batch["images"][0]  # (NV, 3, H, W)
        depths = batch["depths"][0]  # (NV, H, W)
        intrin_invs = batch["intrin_invs"][0]  # (NV, 4, 4)
        extrin_invs = batch["extrin_invs"][0]  # (NV, 4, 4)

        poses = batch["poses"][0]  # (NV, 4, 4)
        focal = batch["focal"][0]  # (1)
        c = batch.get("c")
        # if c is not None:
        #     c = c[0]  # (1)
        NV, _, H, W = images.shape
        cam_rays = utils.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=None if c is None else c[0], ndc=self.cfg.renderer.ndc
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

        test_depths = depths[views_src].unsqueeze(0)  # (1, NS, H, W)
        test_extrin_invs = extrin_invs[views_src].unsqueeze(0)  # (1, NS, 4, 4)
        test_intrin_invs = intrin_invs[views_src].unsqueeze(0)  # (1, NS, 4, 4)

        self.nerf_net.encode(
            test_depths,
            test_extrin_invs,
            test_intrin_invs,
            test_images
        )

        test_rays = test_rays.reshape(1, H * W, -1)
        render_dict = DotMap(self(test_rays, want_weights=True))
        coarse = render_dict.coarse
        fine = render_dict.fine

        using_fine = len(fine) > 0

        alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
        depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)
        rgb_coarse = coarse.rgb[0].reshape(H, W, 3)

        # self.log('rgb min', rgb_coarse_np.min())
        # self.log('rgb max', rgb_coarse_np.max())
        # self.log('al min', alpha_coarse_np.min())
        # self.log('al max', alpha_coarse_np.max())

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
            # self.log('rgb min', rgb_fine_np.min())
            # self.log('rgb max', rgb_fine_np.max())
            # self.log('al min', alpha_fine_np.min())
            # self.log('al max', alpha_fine_np.max())

            depth_fine_cmap = utils.cmap(depth_fine_np) / 255
            alpha_fine_cmap = utils.cmap(alpha_fine_np) / 255
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

    def validation_epoch_end(self, val_outputs):
        out_dir = osp.join(self.cfg.vis_dir, f'epoch_{self.current_epoch:>03d}_step_{self.global_step:>07d}')
        os.makedirs(out_dir, exist_ok=True)

        for item in val_outputs:
            vis = item['vis']
            obj_name = item['obj_name']
            out_name = osp.join(out_dir, f'{obj_name}.png')
            utils.save_img(out_name, (vis * 255).astype(np.uint8))

        all_psnr = torch.cat([x['psnr'].unsqueeze(0) for x in val_outputs])
        all_psnr = self.all_gather(all_psnr)

        self.log('val_psnr', torch.mean(all_psnr), sync_dist=True)

    def train_dataloader(self):
        train_dataset = getattr(dataset, self.dset_cfg.name)(self.dset_cfg, phase='train')
        bs = self.train_cfg.batch_size
        print(f'INFO: Loading {self.dset_cfg.name} train dataset, {len(train_dataset)} objs, bs {bs}')
        data_loader_kwargs = {'num_workers': self.train_cfg.num_workers,
                              'pin_memory': True}
        return DataLoader(train_dataset, batch_size=bs, shuffle=True,
                          **data_loader_kwargs)

    def val_dataloader(self):
        val_dataset = getattr(dataset, self.dset_cfg.name)(self.dset_cfg, phase='val')
        print(f'INFO: Loading {self.dset_cfg.name} validation dataset, {len(val_dataset)} objs')
        print(f'INFO: Validating on {int(len(val_dataset) * self.train_cfg.limit_val_batches)} objs each val epoch')
        data_loader_kwargs = {'num_workers': self.train_cfg.num_workers,
                              'pin_memory': True}
        return DataLoader(val_dataset, batch_size=1, shuffle=False,
                          **data_loader_kwargs)

    def test_dataloader(self):
        val_dataset = getattr(dataset, self.dset_cfg.name)(self.dset_cfg, phase=self.split_to_test)
        print(f'INFO: Loading {self.dset_cfg.name} test dataset, {len(val_dataset)} objs')
        print(f'INFO: Testing on {int(len(val_dataset) * self.limit_test_batches)} objs')
        data_loader_kwargs = {'num_workers': self.train_cfg.num_workers,
                              'pin_memory': True}
        return DataLoader(val_dataset, batch_size=1, shuffle=False,
                          **data_loader_kwargs)

    def test_init(self, args):
        args_dict = vars(args)

        viewlist = args_dict.get('viewlist')
        self.use_source_lut = len(args.viewlist) > 0 if viewlist else False
        if self.use_source_lut:
            print("INFO: Using views from list", viewlist)
            with open(viewlist, "r") as f:
                tmp = [x.strip().split() for x in f.readlines()]
            self.source_lut = {
                x[0] + "/" + x[1]: torch.tensor(list(map(int, x[2:])), dtype=torch.long)
                for x in tmp
            }
        else:
            self.source = torch.tensor(sorted(list(map(int, args.source.split()))), dtype=torch.long)

        self.multicat = args_dict.get('multicat')
        self.free_pose = args_dict.get('free_pose')
        self.include_src = args_dict.get('include_src')

        self.eval_ray_batch_size = args_dict.get('eval_ray_batch_size')

        self.no_compare_gt = args_dict.get('no_compare_gt')
        self.write_depth = args_dict.get('write_depth')
        self.write_compare = args_dict.get('write_compare')

        # prepare output dir
        out_name = osp.basename(self.cfg.work_dir)
        out_name = f'{out_name}_{self.dset_cfg.name}_{args.split}set'
        out_name = f'Debug_{out_name}' if args.debug else out_name
        out_name = f'{out_name}_{args.remarks}' if args.remarks else out_name
        self.test_results_dir = osp.join(args.outroot, out_name)
        os.makedirs(self.test_results_dir, exist_ok=True)

        self.simple_output = True
        self.limit_test_batches = args_dict.get('limit_test_batches')
        self.split_to_test = args_dict.get('split')

        self.args = args

    def test_step(self, batch, batch_idx):
        dpath = batch["path"][0]
        obj_basename = os.path.basename(dpath)
        cat_name = os.path.basename(os.path.dirname(dpath))
        obj_name = cat_name + "_" + obj_basename if self.multicat else obj_basename
        images = batch["images"][0]  # (NV, 3, H, W), [-1, 1]
        depths = batch["depths"][0]  # (NV, H, W)
        intrin_invs = batch["intrin_invs"][0]  # (NV, 4, 4)
        extrin_invs = batch["extrin_invs"][0]  # (NV, 4, 4)

        NV, _, H, W = images.shape

        if self.use_source_lut or self.free_pose:
            if self.use_source_lut:
                obj_id = cat_name + "/" + obj_basename
                source = self.source_lut[obj_id]

            NS = len(source)
            src_view_mask = torch.zeros(NV, dtype=torch.bool)
            src_view_mask[source] = 1

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

        test_images = images[src_view_mask]  # (NS, 3, H, W), [-1, 1]

        test_depths = depths[src_view_mask].unsqueeze(0)  # (1, NS, H, W)
        test_extrin_invs = extrin_invs[src_view_mask].unsqueeze(0)  # (1, NS, 4, 4)
        test_intrin_invs = intrin_invs[src_view_mask].unsqueeze(0)  # (1, NS, 4, 4)

        self.nerf_net.encode(
            test_depths,
            test_extrin_invs,
            test_intrin_invs,
            test_images
        )

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
        obj_out_dir = os.path.join(self.test_results_dir, obj_name)
        os.makedirs(obj_out_dir, exist_ok=True)
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

    def test_epoch_end(self, test_outputs):
        avg_psnr = np.array([x['psnr'] for x in test_outputs]).mean()
        avg_ssim = np.array([x['ssim'] for x in test_outputs]).mean()

        print(f'Final psnr: {avg_psnr}, ssim: {avg_ssim}')
