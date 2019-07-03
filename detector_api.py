import json
import argparse
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, show_single_category_result
import cv2
import glob
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]

class lesion_detector():
    def __init__(self,name='DR_lesion_detector'):
        self.name = name
        self.json_result = None
        self.cfg = None
        self.model = None
        self.threshold = 0.3
    def init_predictor(self,config_dir='',model_dir=''):
        self.cfg = mmcv.Config.fromfile(config_dir)
        self.cfg.model.pretrained = None
        self.model = build_detector(self.cfg.model, test_cfg=self.cfg.test_cfg)
        _ = load_checkpoint(self.model, model_dir)
    def prediction(self,img_dir,show_save_dir=''):
        img = mmcv.imread(img_dir)
        result = inference_detector(self.model, img, self.cfg)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None       
        json_result = dict()
        json_result['image_dir'] = img_dir
        json_result['results']=[]
        
        for label in range(len(bbox_result)):
            bboxes = bbox_result[label]
            for i in range(bboxes.shape[0]):
                if float(bboxes[i][4])> self.threshold:
                    data = dict()
                    data['bbox'] = xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['label'] = str(label+1)
                    json_result['results'].append(data)        

        if not show_save_dir=='':
            image = cv2.imread(img_dir)
            for result in json_result['results']:
                bbox = [int(result['bbox'][0]),int(result['bbox'][1]),int(result['bbox'][2]),int(result['bbox'][3])]
                cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),5)
                cv2.putText(image,str(result['label']),(bbox[0]+bbox[2],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)                
            cv2.imwrite(show_save_dir,image)
            cv2.imshow('test',image)
            cv2.waitKey(0)
        self.json_result = json_result
        return self.json_result
    def getResult(self):
        return self.json_result

def test():
    LesionDetector = lesion_detector()
    config_dir = 'configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1.py'
    model_dir = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4/epoch_9.pth'
    LesionDetector.init_predictor(config_dir,model_dir)
    img_dir = '/data0/qilei_chen/Development/Datasets/KAGGLE_DR/train/4/5304_right.jpeg'
    show_save_dir = '/data0/qilei_chen/Development/test_pytorch_detector.jpg'
    for i in range(100):
        LesionDetector.prediction(img_dir,show_save_dir)

test()