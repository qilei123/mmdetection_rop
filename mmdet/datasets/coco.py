import numpy as np
from pycocotools.coco import COCO
import json
from .custom import CustomDataset

#DATASET = 'DB_4LESIONS'
#DATASET = 'ROP_9LESIONS'
DATASET = 'DB_7LESIONS'
pseudo_threds = 0.0


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]

def py_cpu_nms(dets,scores, thresh):  
    """Pure Python NMS baseline."""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    #scores = dets[:, 4]  #bbox打分
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    #打分从大到小排列，取index  
    order = scores.argsort()[::-1]  
    #keep为最后保留的边框  
    keep = []  
    while order.size > 0:  
    #order[0]是当前分数最大的窗口，肯定保留  
        i = order[0]  
        keep.append(i)  
        #计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= thresh)[0]  
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]  
  
    return keep
WITH_NMS=False

class CocoDataset(CustomDataset):
    
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    
    
    if DATASET=='ROP_2TISSUES':
        CLASSES = ('Macula','OpticDisk')
    if DATASET=='ROP_9LESIONS':
        CLASSES = ('Laser Photocoagulation Spot','artifact','bleeding',
                    'Stage 1: demarcation line','Stage 2: ridge',
                    'Stage 3: ridge with neovascularization',
                    'proliferation','Retina detachment','carcinoma')
    if DATASET=='DB_4LESIONS':
        CLASSES = ('hemorrhages', 'micro-aneurysms', 'hard exudate', 'cotton wool spot')
    if DATASET=='DB_7LESIONS':
        CLASSES = ('1','2','3','4','5','6','7')
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        if 'ROP' in DATASET:
            self.cat2label = {
                cat_id+1: i + 1
                for i, cat_id in enumerate(self.cat_ids)
            }
        else:
            self.cat2label = {
                cat_id: i + 1
                for i, cat_id in enumerate(self.cat_ids)
            }            
        print(self.cat2label)
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def load_Pseudo_annotations(self,Pseudo_ann_file):
        pseudo_ann_info = dict()
        json_pseudo_ann = json.load(open(Pseudo_ann_file))

        for pseudo_ann in json_pseudo_ann:
            pseudo_ann_info[pseudo_ann['image_name']] = pseudo_ann
        return pseudo_ann_info

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def get_Pseudo_ann_info(self, image_name):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_scores = []
        pseudo_ann = self.pseudo_ann_info[image_name]
        for pseudo_box in pseudo_ann['box_results']:
            if pseudo_box['score']>pseudo_threds:
                x1, y1, w, h = pseudo_box['bbox']
                box = [int(x1), int(y1), int(x1 + w - 1), int(y1 + h - 1)]
                gt_bboxes.append(box)
                gt_labels.append(pseudo_box['category_id'])
                gt_scores.append(pseudo_box['score'])
        if gt_bboxes:
            if WITH_NMS:
                print(len(gt_bboxes))
                temp_gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                #temp_gt_labels = np.array(gt_labels, dtype=np.int64)
                temp_gt_scores = np.array(gt_scores,dtype = np.float32)

                indexes = py_cpu_nms(temp_gt_bboxes,temp_gt_scores,0.15)
                temp_gt_bboxes = []
                temp_gt_labels = []
                for index in indexes:
                    temp_gt_bboxes.append(gt_bboxes[int(index)])
                    temp_gt_labels.append(gt_labels[int(index)])
                gt_bboxes=temp_gt_bboxes    
                gt_labels=temp_gt_labels
                print(len(gt_bboxes))
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)        
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        pseudo_ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        return pseudo_ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                if 'ROP' in DATASET:
                    gt_labels.append(self.cat2label[ann['category_id']+1])
                else:
                    gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann
