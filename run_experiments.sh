export CUDA_VISIBLE_DEVICES=0
config_file=configs/single_stage/ssd512_coco_dr_4lesions.py
sh tools/dist_train_DR_4lesions.sh ${config_file} 1
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_kld_loss.py
#sh tools/dist_train_DR_4lesions_3000.sh ${config_file} 2
