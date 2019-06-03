import argparse
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, show_single_category_result
import cv2
import glob
import os

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
        '--score_thr', default=0.1,type = float,
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

# construct the model and load checkpoint
#model_dir = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model_dir = args.model_dir
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, model_dir)

print(model)
# test a single 
resize_scale = args.resize_scale
img_dir = args.img_dir
img = mmcv.imread(img_dir)
height, width, depth = img.shape
img = cv2.resize(img,(int(resize_scale*width),int(resize_scale*height)))
result = inference_detector(model, img, cfg)
show_single_category_result(img, result,score_thr = args.score_thr,category_id=args.single_category_id,out_file='/data0/qilei_chen/Development/show_single_label_result.jpg')
show_result(img, result,score_thr = args.score_thr,out_file='/data0/qilei_chen/Development/show_result.jpg')
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