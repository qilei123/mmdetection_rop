import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/mask_rcnn_x101_64x4d_fpn_1x_2tissues.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
#model_dir = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model_dir = '../2TISSUES/mask_epoch_12.pth'
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, )

# test a single image
img_dir = '/media/cql/DATA1/data/2TISSUES/val2014/val_3_2631ccbad6435fa1dbbcdc4c38b8d5fb.png'
img = mmcv.imread(img_dir)
result = inference_detector(model, img, cfg)
show_result(img, result)

'''
# test a list of images
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result)
'''