import mmcv

from . import assigners, samplers
import torch
import torch.nn as nn

def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(
            cfg, assigners, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(
            cfg, samplers, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg,pseudo_bboxes = None):

    bbox_assigner = build_assigner(cfg.assigner)

    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels)

    if not pseudo_bboxes is None:
        union_bboxes = torch.cat((gt_bboxes,pseudo_bboxes),0)
        #union_labels = torch.cat((gt_labels,pseudo_labels),0)
        union_assign_result = bbox_assigner.assign(
            bboxes, union_bboxes, gt_bboxes_ignore,
            None)
        with_union=True
    else:
        union_assign_result=None
        with_union=False
    
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes,
                                          gt_labels,                    
                                          union_assign_result=union_assign_result,
                                          with_union=with_union,)
    return assign_result, sampling_result
