export CUDA_VISIBLE_DEVICES=0,1
config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_2000_v2_with_focal_loss_smallset.py
sh tools/dist_train_DR_4lesions_3000.sh ${config_file} 2
