from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.models.transformer import Transformer
from lib.models.backbone import build_backbone

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PoseTransformer(nn.Module):

    def __init__(self, cfg, backbone, transformer, **kwargs):
        super(PoseTransformer, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.num_queries = 256
        self.transformer = transformer
        self.backbone = backbone
        hidden_dim = transformer.d_model
        # self.class_embed_cofw = nn.Linear(hidden_dim, 29 + 1)
        self.class_embed_wflw = nn.Linear(hidden_dim, 68 + 57)
        # self.class_embed_300w = nn.Linear(hidden_dim, 68 + 1)
        self.sp_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(156, hidden_dim)
        self.aux_loss = extra.AUX_LOSS

        self.num_feature_levels = extra.NUM_FEATURE_LEVELS
        if self.num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
            # 初始化
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(68 + 57) * bias_value
            nn.init.constant_(self.kpt_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.kpt_embed.layers[-1].bias.data, 0)
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            num_pred = transformer.decoder.num_layers  ##解码器的层数
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])  ## 每一个解码层都添加全连接网络进行预测
            self.kpt_embed = nn.ModuleList([self.kpt_embed for _ in range(num_pred)])
        else:
            # self.input_proj = nn.Conv2d(self.backbone.num_channels[0], hidden_dim, 1)
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.backbone.num_channels[ii], hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim)) for
                                             ii in range(len(self.backbone.num_channels))])

    def forward(self, x):
        # src, pos = self.backbone(x)

        src, poses = self.backbone(x)
        srcs = []
        for l, feat in enumerate(src):
            srcs.append(self.input_proj[l](feat))
        src_flatten = []
        pos_flatten = []
        for lvl, (src, pos) in enumerate(zip(srcs, poses)):
            src = src.flatten(2).transpose(1, 2)
            pos = pos.flatten(2).transpose(1, 2)
            src_flatten.append(src)
            pos_flatten.append(pos)

        src_flatten = torch.cat(src_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 1)

        # output_wflw = self.transformer(self.input_proj(src[-1]), None, self.query_embed.weight, pos[-1])

        hs, sp_output = self.transformer(src_flatten, None, self.query_embed.weight, pos_flatten)

        outputs_sp = self.sp_head(sp_output[-1]).view(sp_output.shape[1], sp_output.shape[2], 16, 16)
        outputs_class = self.class_embed_wflw(hs)
        outputs_coord = self.kpt_embed(hs).sigmoid()
        #'pred_logits' [bs, 156, 125], 156为查询的数量，125对应为背景类， 'pred_coords': [bs, 156, 2]
        out_wflw = {
            'pred_logits': outputs_class[-1],
            'pred_coords': outputs_coord[-1],
            'pred_sp': outputs_sp
            }

        if self.aux_loss:
            out_wflw['aux_outputs'] = self._set_aux_loss(
                outputs_class,
                outputs_coord)
        #     outs.append(out)
        return out_wflw

    @torch.jit.unused
    def _set_aux_loss(self,
                      outputs_class,
                      outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def get_face_alignment_net(cfg, is_train, **kwargs):
    extra = cfg.MODEL.EXTRA

    transformer = Transformer(d_model=extra.HIDDEN_DIM, dropout=extra.DROPOUT, nhead=extra.NHEADS,
                              dim_feedforward=extra.DIM_FEEDFORWARD,
                              num_encoder_layers=extra.ENC_LAYERS, num_decoder_layers=extra.DEC_LAYERS,
                              normalize_before=extra.PRE_NORM,
                              return_intermediate_dec=True,
                              num_clusters=extra.NUM_CLUSTERS,)
    pretrained = is_train and cfg.MODEL.INIT_WEIGHTS
    backbone = build_backbone(cfg, pretrained)
    model = PoseTransformer(cfg, backbone, transformer, **kwargs)

    return model
