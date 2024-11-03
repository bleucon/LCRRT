# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from https://github.com/tianmaxingkong12/face-alignment-transformer
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class Face300W(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET_300W.TRAINSET
        else:
            self.csv_file = cfg.DATASET_300W.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.output_size = cfg.MODEL.IMAGE_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET_300W.SCALE_FACTOR
        self.rot_factor = cfg.DATASET_300W.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET_300W.FLIP
        self.data_root = cfg.DATASET_300W.DATA_DIR

        # load annotations
        landmarks_frame = pd.read_csv(self.csv_file, sep="\t")
        extended_df = pd.concat([landmarks_frame] * (20000 // len(landmarks_frame) + 1), ignore_index=True)
        self.landmarks_frame = extended_df.iloc[:20000]

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        d = 'trainset/' if self.is_train else 'ibug/'

        image_path = self.data_root + self.landmarks_frame.iloc[idx, 0]
        scale = self.landmarks_frame.iloc[idx, 3]

        center_w = self.landmarks_frame.iloc[idx, 4]
        center_h = self.landmarks_frame.iloc[idx, 5]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 2]
        # pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = np.array(list(map(float, pts.split(","))), dtype=np.float32).reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        heatmap_target = np.zeros((nparts, self.heatmap_size[0], self.heatmap_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                heatmap_target[i] = generate_target(heatmap_target[i],
                                                    (tpts[i]-1)*self.heatmap_size/self.input_size,
                                                    self.sigma,)

        # ---------------------- update coord target -------------------------------
        target = tpts[:, 0:2] / self.input_size[0]

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        heatmap_target = torch.Tensor(heatmap_target)

        target_weight = np.ones((nparts, 1), dtype=np.float32)
        target_weight = torch.from_numpy(target_weight)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'rotate': r, 'pts': torch.Tensor(pts), 'tpts': tpts,
                'img_pth': image_path}

        return img, target, heatmap_target, target_weight, meta


if __name__ == '__main__':

    pass
