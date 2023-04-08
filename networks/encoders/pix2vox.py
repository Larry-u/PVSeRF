# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=4)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 6, 6])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 128, 4, 4])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 128, 4, 4])
        return image_features


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=cfg.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_voxels = []
        raw_features = []

        for features in image_features:
            gen_voxel = features.view(-1, 256, 2, 2, 2)
            # print(gen_voxel.size())   # torch.Size([batch_size, 256, 2, 2, 2])
            gen_voxel = self.layer1(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 128, 4, 4, 4])
            gen_voxel = self.layer2(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 64, 8, 8, 8])
            gen_voxel = self.layer3(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_voxel = self.layer4(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            raw_feature = gen_voxel
            gen_voxel = self.layer5(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            raw_feature = torch.cat((raw_feature, gen_voxel), dim=1)
            # print(raw_feature.size()) # torch.Size([batch_size, 9, 32, 32, 32])

            gen_voxels.append(torch.squeeze(gen_voxel, dim=1))
            raw_features.append(raw_feature)

        gen_voxels = torch.stack(gen_voxels).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_voxels.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_voxels


class Pix2VoxF(torch.nn.Module):
    def __init__(self, cfg):
        super(Pix2VoxF, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, rendering_images):
        image_features = self.encoder(rendering_images)
        raw_features, generated_volume = self.decoder(image_features)

        return raw_features, generated_volume


class Encoder64(Encoder):
    def __init__(self, cfg):
        super(Encoder64, self).__init__(cfg)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
            # torch.nn.MaxPool2d(kernel_size=4)  # remove maxpool in 64
        )

class Encoder128(Encoder):
    def __init__(self, cfg):
        super(Encoder128, self).__init__(cfg)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
            # torch.nn.MaxPool2d(kernel_size=4)
        )


class Pix2VoxF64(Pix2VoxF):
    def __init__(self, cfg):
        super(Pix2VoxF64, self).__init__(cfg)
        self.encoder = Encoder64(cfg)  # change encoder


class Pix2VoxF128(Pix2VoxF):
    def __init__(self, cfg):
        super(Pix2VoxF128, self).__init__(cfg)
        self.encoder = Encoder128(cfg)  # change encoder