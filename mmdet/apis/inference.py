import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
import types

def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, cfg, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)


def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device)


def show_result(img, result, dataset='coco', score_thr=0.3, out_file=None,show=True,win_name='DB'):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show = show,
        out_file=out_file,
        win_name=win_name,
        wait_time=0)

def show_single_category_result(img, result, dataset='coco', score_thr=0.3, out_file=None,category_id=0):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    single_category_indexes = []
    category_index = 0
    for label in labels:
        if label==category_id:
            single_category_indexes.append(category_index)
        category_index+=1
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes[single_category_indexes,:],
        labels[single_category_indexes],
        class_names=class_names,
        score_thr=score_thr,
        out_file=out_file)

def show_result_rop_2tissue(img, result, dataset='coco', score_thr=0.3, out_file=None):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    print(bboxes)
    print(labels)
    print(type(bboxes))
    h0_score = 0
    index_0 = -1
    h1_score = 0
    index_1 = -1

    for i in range(bboxes.shape[0]):
        if labels[i]==0:
            if bboxes[i,4]>h0_score:
                h0_score = bboxes[i,4]
                index_0 = i
        else:
            if bboxes[i,4]>h1_score:
                h1_score = bboxes[i,4]
                index_1 = i
    indexes = []
    if index_0!=-1:
        indexes.append(index_0)
    if index_1!=-1:
        indexes.append(index_1)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes[indexes,:],
        labels[indexes],
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None)
