
# -*- coding:utf-8 -*-
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
import datetime
import numpy as np

def color_libs(index=0):
    clibs = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
    return clibs[index]
    
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

def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep

def nms_result(json_result):
    boxes  = []
    boxscores = []
    for result in json_result['results']:
        boxes.append([int(result['bbox'][1]),int(result['bbox'][0]),int(result['bbox'][1])+int(result['bbox'][3]),int(result['bbox'][0])+int(result['bbox'][2])])
        boxscores.append(result['score'])
    boxes = np.array(boxes,dtype = np.float32)
    boxscores = np.array(boxscores,dtype = np.float32)
    #print(boxes)
    if len(boxes)>0:
        #index = py_cpu_softnms(boxes, boxscores, method=3)
        indexes = py_cpu_nms(boxes,boxscores,0.15)
        #print(index)
        temp_list = []
        for index in indexes:
            temp_list.append(json_result['results'][int(index)])
        json_result['results']=temp_list
import os
def show_results(json_result_dir,image_folder,save_folder,score_threshold=0.3):

    json_results = json.load(open(json_result_dir))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    count = 0
    count_f = 0
    for image_result in json_results['results']:
        count+=1
        #print(image_result)
        if len(image_result['results'])>0:
        #if len(image_result['box_results'])>0:
            image_loaded = False
            #for box_result in image_result['box_results']:
            for box_result in image_result['results']:
                if box_result['score']>=score_threshold:
                    
                    if image_loaded==False:
                        image_loaded = True
                        image = cv2.imread(image_result['image_dir'])
                        count_f += 1
                    bbox = box_result['bbox']
                    cv2.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),(0,255,0),2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image,str(box_result['label']),(int(bbox[0]+bbox[2]),int(bbox[1])), font, 1,(0,255,0),2,cv2.LINE_AA)
            if image_loaded:
                image_name = os.path.basename(image_result['image_dir'])
                cv2.imwrite(os.path.join(save_folder,image_name.replace('.jpeg','_show.jpeg')),image)
                #os.system('cp '+image_result['image_dir']+' '+save_folder)
                pass
        if count%1000==0:
            print(str(count/1000)+'k')

    print(count)
    print(count_f)

def show_groundtruth(json_annotation_dir,image_folder,save_folder):
    json_annotation = json.load(open(json_annotation_dir))
    images_dict = dict()

    for img in json_annotation['images']:
        images_dict[img['id']] = img

    annos_by_image_id = dict()
    for anno in json_annotation['annotations']:
        if not anno['image_id'] in annos_by_image_id:
            annos_by_image_id[anno['image_id']]=[]
        annos_by_image_id[anno['image_id']].append(anno)

    for key in annos_by_image_id.keys():
        image = cv2.imread(os.path.join(image_folder,images_dict[key]['file_name']))
        for anno in annos_by_image_id[key]:
            bbox = [int(anno['bbox'][0]),int(anno['bbox'][1]),int(anno['bbox'][2]),int(anno['bbox'][3])]
            cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),5)
            cv2.putText(image,str(anno['category_id']),(bbox[0]+bbox[2],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_folder,images_dict[key]['file_name']),image)
        '''
        view_scale = 0.4
        image = cv2.resize(image,(int(view_scale*image.shape[1]),int(view_scale*image.shape[0])))
        cv2.imshow('gt_show',image)
        cv2.waitKey(0)
        '''
class lesion_detector():
    def __init__(self,name='DR_lesion_detector'):
        self.name = name
        self.json_result = None
        self.cfg = None
        self.model = None
        self.threshold = 0.0
    def init_predictor(self,config_dir='/home/ubuntu/qilei_chen/mmdetection_rop/configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1.py',model_dir='/home/ubuntu/qilei_chen/mmdetection_models/epoch_9.pth'):
        self.cfg = mmcv.Config.fromfile(config_dir)
        self.cfg.model.pretrained = None
        self.model = build_detector(self.cfg.model, test_cfg=self.cfg.test_cfg)
        _ = load_checkpoint(self.model, model_dir)
    def prediction(self,img_dir,show_save_dir='/home/ubuntu/Code/AI_4_Retinaimage/result_img/pytorch_detector_result.jpg'):
        #oldtime=datetime.datetime.now()
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
        nms_result(json_result)
        if not show_save_dir=='':
            image = cv2.imread(img_dir)
            for result in json_result['results']:
                bbox = [int(result['bbox'][0]),int(result['bbox'][1]),int(result['bbox'][2]),int(result['bbox'][3])]
                color = color_libs(int(result['label']))
                cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color,2)
                cv2.putText(image,str(result['label']),(bbox[0]+bbox[2],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,color,2,cv2.LINE_AA)                
            cv2.imwrite(show_save_dir,image)
            #cv2.imshow('test',image)
            #cv2.waitKey(0)
        json_result['result_img_dir'] = show_save_dir
        self.json_result = json_result
        #newtime=datetime.datetime.now()
        #print("detector prediction time:")
        #print((newtime-oldtime).microseconds/1000)
        #print(json_result)
        return self.json_result
    def getResult(self):
        return self.json_result
    def getDetectorName(self):
        return self.name
import glob

DEPLOY_TEST=False

def test():
    LesionDetector = lesion_detector()
    if DEPLOY_TEST:
        config_dir = '/home/ubuntu/qilei_chen/mmdetection_rop/configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1.py'
        model_dir = '/home/ubuntu/qilei_chen/mmdetection_models/epoch_9.pth'
        img_dir = '/home/ubuntu/test/*.jpg'
        show_save_dir = '/home/ubuntu/Code/AI_4_Retinaimage/result_img/test_pytorch_detector.jpg'
    else:
        #config_dir = '/data0/qilei_chen/Development/mmdetection_rop/configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1.py'
        #config_dir = '/data0/qilei_chen/Development/mmdetection_rop/configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_with_pseudo_gt_v1_with_nms.py'
        config_dir = 'C:/Users/Administrator/Desktop/Development/mmdetection_rop/configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_with_pseudo_gt_v1_with_nms.py'
        #config_dir = 'C:/Users/Administrator/Desktop/Development/mmdetection/configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1.py'
        #config_dir = 'C:/Users/Administrator/Desktop/Development/mmdetection/configs/faster_rcnn_x101_32x4d_fpn_1x.py'
        #model_dir = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_deephead_v1/epoch_9.pth'
        #model_dir = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_head_v1_with_pseudo_gt_v1_with_nms/epoch_9.pth'
        model_dir = 'C:/Users/Administrator/Desktop/Development/models/epoch_9.pth'
        #img_dir = '/data0/qilei_chen/AI_EYE/*.jpg'
        #img_dir = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/train2014/*.jpg'
        #img_dir = '/data0/qilei_chen/Development/Debug/*.jpg'
        img_dir = 'C:/Users/Administrator/Desktop/Development/images/*.jpg'
        #show_save_dir = '/data0/qilei_chen/Development/test_pytorch_detector.jpg'
        show_save_dir = 'C:/Users/Administrator/Desktop/Development//test_pytorch_detector.jpg'
    LesionDetector.init_predictor(config_dir,model_dir)
    #show_save_dir = ''
    img_dirs = glob.glob(img_dir)
    #for i in range(10000):
    results = dict()
    results['results']=[]
    for img_dir in img_dirs:
        print(img_dir)
        oldtime=datetime.datetime.now()
        result = LesionDetector.prediction(img_dir,show_save_dir)
        newtime=datetime.datetime.now()
        print((newtime-oldtime).microseconds/1000)
        results['results'].append(result)
    #with open('/data0/qilei_chen/AI_EYE/BostonAI4DB7/head_v1_detect_results1.json','w') as json_file:
    #    json.dump(results,json_file)

def test_show_results():
    #json_result_dir = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/train2014_head_v1_results.json'
    json_result_dir = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/head_v1_detect_results1.json'
    image_folder = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/train2014'
    #save_folder = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/train2014_head_v1_results_show'
    save_folder = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/train2014_head_v1_results_show1'
    show_results(json_result_dir,image_folder,save_folder)

def test_show_gt():
    json_annotation_dir = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/annotations/instances_train2014.json.original.opt_v2.hflip'
    image_folder = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/train2014_head_v1_results_show'
    save_folder = '/data0/qilei_chen/AI_EYE/BostonAI4DB7/train2014_head_v1_results_show'
    show_groundtruth(json_annotation_dir,image_folder,save_folder)
if __name__ == "__main__":
    test()
    #test_show_results()
    #test_show_gt()