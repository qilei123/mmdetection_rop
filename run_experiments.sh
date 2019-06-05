export CUDA_VISIBLE_DEVICES=0,1
config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_7lesions.py
sh tools/dist_train_DR_4lesions.sh ${config_file} 2
