import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from .anchor_head import AnchorHead
from ..registry import HEADS
from torch.autograd import Variable


def decode(head, output_channel):
    decode_conv = nn.Sequential(
        nn.Conv2d(head, 32, kernel_size=1, stride=1),
        #nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, output_channel, kernel_size=1, stride=1)
    )
    return decode_conv
def encode(head, h):
    conv1 = nn.Conv2d(head, h, kernel_size=3, stride=1, padding=1)
    return conv1

@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

        if self.use_kl_loss:
            self.encode_conv = encode(self.feat_channels,10)
            self.decode_conv = decode(5,self.feat_channels)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        ##### add a relative location feature map
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        
        #conv1*1(x)

        rpn_cls_score = self.rpn_cls(x)
        if self.use_kl_loss:
            x = self.encode_conv(x)
            x,mu,logvar = self._reparameterization(x)
            x = self.decode_conv(x)

        rpn_bbox_pred = self.rpn_reg(x)
        if self.use_kl_loss:
            return rpn_cls_score, rpn_bbox_pred, mu, logvar
        else:
            return rpn_cls_score,rpn_bbox_pred, None,None

    def loss(self,
             cls_scores,
             bbox_preds,
             mus,
             logvars,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             pseudo_bboxes = None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            mus,
            logvars,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore,
            pseudo_bboxes=pseudo_bboxes)
        if self.use_kl_loss:
            return dict(
                loss_rpn_cls=losses['loss_cls'], loss_rpn_reg=losses['loss_reg'],loss_kld=losses['loss_kld'])
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_reg=losses['loss_reg'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

    def _reparameterization(self, input):
        half_size = int(input.size(1)/2)
        #print(half_size)
        mu = input[:,:half_size]
        if self.train:
            logvar = input[:,half_size:]
            std = torch.exp(0.5*logvar)
            eps = Variable(torch.cuda.FloatTensor(std.shape).normal_(), requires_grad=False)
            return mu + eps * std,mu,logvar
        else:
            # when testing, propagate mu directly
            return mu
    '''
    def decode(self,head, output_channel):
        decode_conv = nn.Sequential(
            nn.Conv2d(head, 32, kernel_size=1, stride=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, output_channel, kernel_size=1, stride=1)
        )
        return decode_conv
    def encode(self,head, h):
        conv1 = nn.Conv2d(head, h, kernel_size=3, stride=1, padding=1)

        return conv1
    '''