git pull
img_dir=/data0/qilei_chen/AI_EYE/BostonAI4DB7/val2014/flip_val_675_ZJEYY01-S40rec40-03-ddef6826-6ecb-4ac5-a69f-d83c4da28f3b.jpg
config_dir=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_2000_v2_with_focal_loss_smallset.py
model_dir=/data0/qilei_chen/AI_EYE/BostonAI4DB7/work_dirs/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_2000_v2_with_focal_loss_smallset/epoch_10.pth
python3 inference_single_img.py --img_dir ${img_dir} --config_dir ${config_dir} --model_dir ${model_dir}
