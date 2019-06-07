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
save_dir = '/data0/qilei_chen/Development/show_test/'
def cutMainROI1(img,folder):
    #x=img[img.shape[0]/2,:,:].stum(1)
    #xx = img[img.shape[0]/2,:,:]
    #yy = img[:,img.shape[1]/2,:]
    w = img.shape[1]
    h = img.shape[0]
    x_s = 0
    x_e = 0
    threshold = 10
    for i in range(w):
        if not (img[int(h/2)][i][0]<10 and img[int(h/2)][i][1]<10 and img[int(h/2)][i][2]<10):
            x_s = i
            break 
    
    for i in range(w):
        if not (img[int(h/2)][w-i-1][0]<10 and img[int(h/2)][w-i-1][1]<10 and img[int(h/2)][w-i-1][2]<10):
            x_e = w-i
            break 
    y_s = 0
    y_e = 0
    for i in range(h):
        if not (img[i][int(w/2)][0]<10 and img[i][int(w/2)][1]<10 and img[i][int(w/2)][2]<10):
            y_s = i
            break 

    for i in range(h):
        if not (img[h-i-1][int(w/2)][0]<10 and img[h-i-1][int(w/2)][1]<10 and img[h-i-1][int(w/2)][2]<10):
            y_e = h-i
            break
    #print 'new image roi:'+str([y_s,y_e,x_s,x_e])
    
    cut_img = img[int(y_s):int(y_e),int(x_s):int(x_e)]
    
    if y_e-y_s<100 or x_e-x_s<100:
        cut_img = img

    cv2.imwrite(save_dir+folder+'_cropped_img.jpg',cut_img)
    #cv2.imshow('test',cut_img)
    #cv2.waitKey(0)
    return cut_img,x_s,y_s

def parse_args():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument(
        '--img_dir', default='/media/cql/DATA1/data/dr_2stages_samples/wrong_samples/0/297_left.jpeg', 
        help='image path for testing')
    parser.add_argument(
        '--config_dir', default='configs/mask_rcnn_x101_64x4d_fpn_1x_2tissues.py',
        help='config file for testing')
    parser.add_argument(
        '--model_dir', default='../2TISSUES/mask_epoch_12.pth',
        help='model file for testing')
    parser.add_argument(
        '--score_thr', default=0.3,type = float,
        help='score threshold for testing')
    parser.add_argument(
        '--resize_scale', default=1,type = float,
        help='resize scale for testing')
    parser.add_argument(
        '--single_category_id', default=0,type = int,
        help='single category for testing')
    args = parser.parse_args()
    return args

args = parse_args()

cfg = mmcv.Config.fromfile(args.config_dir)
cfg.model.pretrained = None

#img = cutMainROI1(cv2.imread(args.img_dir))
# construct the model and load checkpoint
#model_dir = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model_dir = args.model_dir
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, model_dir)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

print(model)

model.backbone.maxpool.register_forward_hook(get_activation('conv1'))

# test a single 

resize_scale = args.resize_scale

folders = ['4']
dataset_dir = '/data0/qilei_chen/AI_EYE/kaggle_data/dataset_4stages/train_4/'
#folders = ['0']
#dataset_dir = '/data0/qilei_chen/AI_EYE/kaggle_data/train_binary/'
img_set = 'test'
for folder in folders:
    img_dirs = glob.glob(dataset_dir+folder+'/*.jpeg')
    for img_dir in img_dirs:
        #print(img_dir)
        
        img_file_name = os.path.basename(img_dir)
        output_file=save_dir+img_set+'/'+folder+'/'+img_file_name
        #img_dir = args.img_dir
        if not os.path.exists(output_file):
            img = cv2.imread(img_dir)
        
            img = cutMainROI1(img,folder)

            #cv2.imwrite(save_dir+'cropped_img.jpg',img)
            img = mmcv.imread(save_dir+folder+'_cropped_img.jpg')
            height, width, depth = img.shape
            img = cv2.resize(img,(int(resize_scale*width),int(resize_scale*height)))
            result = inference_detector(model, img, cfg)
            '''
            act_gpu = activation['conv1'].squeeze()
            act = act_gpu.cpu().numpy()
            print(act.shape)
            fig, axarr = plt.subplots(act.shape[0])
            for idx in range(act.shape[0]):
                cv2.imshow('test',act[idx,:,:])
                cv2.waitKey(0)
            '''
            '''
            show_single_category_result(img, result,score_thr = args.score_thr,
                category_id=args.single_category_id,
                out_file=save_dir+str(time.time())+'_show_single_label_result.jpg')
            '''    
            
            show_result(img, result,score_thr = args.score_thr,
                out_file=output_file,show=False)
'''
folder = '/media/cql/DATA0/Development/RetinaImg/dataset/IDRID/C. Localization/1. Original Images/b. Testing Set'
resize_scale = 0.2
paths = glob.glob(os.path.join(folder,'*.jpg'))
for path in paths:
    img = mmcv.imread(path)
    height, width, depth = img.shape
    img = cv2.resize(img,(int(resize_scale*width),int(resize_scale*height)))
    result = inference_detector(model, img, cfg)
    show_result(img, result,score_thr = args.score_thr)
'''
'''
# test a list of images
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result)
'''