export CUDA_VISIBLE_DEVICES=0
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset.py
#sh tools/dist_train_DR_4lesions_3000.sh ${config_file} 1
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset2.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_baseline.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_second_round_v2.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_2000_v2.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_hight_neg_iou_thres.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_hight_neg_iou_thres_lower1_pos_iou_thres.py
config_file=configs/mask/mask_rcnn_x101_64x4d_fpn_1x_ridge_in_one.py
sh tools/dist_train_DR_4lesions_3000.sh ${config_file} 1
config_file=configs/mask/mask_rcnn_x101_64x4d_fpn_1x_ridge_in_one_with_focal_loss.py
sh tools/dist_train_DR_4lesions_3000.sh ${config_file} 1
