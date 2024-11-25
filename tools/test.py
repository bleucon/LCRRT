# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import os
import pprint
import time
import torch
import torch.nn.parallel
from torchinfo import summary
# import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
# import torchvision.transforms as transforms
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config as cfg
from lib.config import update_config
from lib.models.matcher import build_matcher
from lib.core.loss import SetCriterion
from lib.core.function import validate
from lib.utils.utils import create_logger, model_key_helper

from lib.dataset import Face300W, COFW, AFLW, WFLW
import lib.models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment cfgure file name',
                        type=str,
                        default="./wflw.yaml")

    parser.add_argument('opts',
                        help="Modify cfg options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default="checkpoint_0.06632565202389999.pth")
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('lib.models.'+cfg.MODEL.NAME+'.get_face_alignment_net')(
        cfg, is_train=False
    ).cuda()

    if args.modelDir:
        model_state_file = args.modelDir
    else:
        model_state_file = os.path.join(
            final_output_dir, cfg.TRAIN.CHECKPOINT
        )
    logger.info('=> loading model from {}'.format(model_state_file))
    state = torch.load(model_state_file)

    last_epoch = state['epoch']
    best_nme = state['best_nme']
    if 'best_state_dict' in state.keys():
        state = state['best_state_dict']
    model.load_state_dict(state['state_dict'].state_dict())
    # model.load_state_dict(state['state_dict'].module.state_dict())

    # 计算模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total parameters: {total_params / 1e6:.2f}M')
    summary(model, input_size=(1, 3, 512, 512))
    # logger.info(f"Total FLOPs: {flops.total()}")  # 输出总 FLOPs

    # define loss function (criterion) and optimizer
    matcher = build_matcher(cfg.MODEL.LANDMARKS)
    weight_dict = {'loss_ce': 1, 'loss_kpts': cfg.MODEL.EXTRA.KPT_LOSS_COEF, 'map_loss': 1}
    criterion = SetCriterion(cfg.MODEL.LANDMARKS, matcher, weight_dict, cfg.MODEL.EXTRA.EOS_COEF,
                             ['labels', 'kpts', 'cardinality', 'sp_map'])
    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )

    # valid_dataset = WFLW(
    #     cfg,
    #     # cfg.DATASET.ROOT, cfg.DATASET.TEST_SET,
    #     False,
    #     # transforms.Compose([
    #     #     transforms.ToTensor(),
    #     #     normalize,
    #     # ])
    # )

    val_300W_loader = DataLoader(
        dataset=Face300W(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    val_wflw_loader = DataLoader(
        dataset=WFLW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    val_aflw_loader = DataLoader(
        dataset=AFLW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    val_cofw_loader = DataLoader(
        dataset=COFW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    # val_loaders = {"WFLW": valid_loader}
    val_loaders = {'300W': val_300W_loader, 'WFLW': val_wflw_loader, 'COFW': val_cofw_loader, 'AFLW': val_aflw_loader}
    # evaluate on validation set
    validate(cfg, val_loaders, model, criterion,
             last_epoch, time.time(), None)

if __name__ == '__main__':
    main()