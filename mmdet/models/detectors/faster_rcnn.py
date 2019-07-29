from .two_stage import TwoStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 with_pseudo_gt_at_rpn=False,
                 with_pseudo_gt_at_rcnn=False):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            with_pseudo_gt_at_rpn=with_pseudo_gt_at_rpn,
            with_pseudo_gt_at_rcnn=with_pseudo_gt_at_rcnn)
