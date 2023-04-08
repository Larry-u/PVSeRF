# PVSeRF
The official repository of our ACM MM 2022 paper: "PVSeRF: Joint Pixel-, Voxel- and Surface-Aligned Radiance Field for Single-Image Novel View Synthesis". 

If you find this repository useful, please consider citing our paper:grinning:

## Prerequisite
 - [Pytorch](https://pytorch.org/get-started/locally/) (==1.8.0)

 - [Pytorch-lightning](https://www.pytorchlightning.ai/index.html) (==1.4.6)
 - [yacs](https://github.com/Larry-u/yacs.git) (my modified version)

 - cudatoolkit==11.1

 - numpy

 - matplotlib

 - opencv-python

It is recommended to install prerequisites using virtual environment:

```
# using conda
conda env create -f environment.yaml

# using pip
pip install -r requirements.txt
```


## Dataset
We use the same datasets and splits from [pixel-nerf](https://alexyu.net/pixelnerf/), you can download the ShapeNet 64x64 dataset (from NMR) for category-agnostic experiment and the SRN chair/car dataset for category-specific experiment from pixel-nerf's [repo](https://github.com/sxyu/pixel-nerf).

Then unzip the datasets and put them under the `data/` directory. The folder structure will be as follows:

```
project_root/
    data/
        NMR_dataset/
        src_cars/
        src_chairs/
```

## Training

### Pretrain the volume generator $G_v$

We use [Pix2Vox](https://github.com/hzxie/Pix2Vox) as the volume generator backbone, you can follow the procedures from it to pretrain $G_v$. Since the original Pix2Vox is trained on `224x224` resolution, you have to slightly modify the network architecture to fit ShapeNet's `64x64` resolution and SRN's `128x128` resolution. More details are referred to `networks/encoders/pix2vox.py`.

### Pretrain the point set generator $G_s$

For point set generator, we use [GraphX-convolution](https://github.com/justanhduc/graphx-conv) as backbone. Instead of regressing point set in camera space, we regress the world-space point set. To prepare the training data for GraphX-convolution, for NMR dataset, you can subsample `N` points from the `pointcloud.npz` of each object; for SRN data, you have to sample `N` points from the mesh obj file of each object (for reference, we use the `sample_surface` fuction from `trimesh`).

After the preparation of GT point cloud, you can train a point set generator following the steps in [GraphX-convolution](https://github.com/justanhduc/graphx-conv).

### Jointly train PVSeRF
Firtly, you have to replace the path of `vox_encoder_ckpt` and `graphx_ckpt` in config yaml file, repectively. Then, run the following command to train PVSeRF:

```
# train on single GPU
python train.py -c configs/pvserf.yaml

# train on multiple GPUs (using DistributedDataParallel)
python train.py -c configs/pvserf.yaml --trainer.gpus 8 --acc ddp

# optional arguments
--trainer.fast_dev_run    # if setting this to true, the trainer will run a full training cycle (a single train&val iteration)
```

## Testing
You can use the following command to test a trained PVSeRF model:
```
# use 4 GPUs, 4x faster
python eval.py --trainer.gpus 4 -L viewlist/src_dvr.txt --multicat --no_compare_gt --write_compare \
-r path/to/checkpoints.ckpt

# optional
--vis_geometry
```

### Acknowledgement
Thanks for the public code from [pixel-nerf](https://alexyu.net/pixelnerf/), [Pix2Vox](https://github.com/hzxie/Pix2Vox), and [GraphX-convolution](https://github.com/justanhduc/graphx-conv). I also appreciate the excellent PyTorch research framework from [Lightning](https://www.pytorchlightning.ai/index.html).

### Citation
```
@inproceedings{yu2022pvserf,
  title={PVSeRF: joint pixel-, voxel-and surface-aligned radiance field for single-image novel view synthesis},
  author={Yu, Xianggang and Tang, Jiapeng and Qin, Yipeng and Li, Chenghong and Han, Xiaoguang and Bao, Linchao and Cui, Shuguang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1572--1583},
  year={2022}
}
```



 


        
