export CUDA_VISIBLE_DEVICES=0,1
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset.py
#sh tools/dist_train_DR_4lesions_3000.sh ${config_file} 1
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset2.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_baseline.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v2.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_InstanceBalancedPosSampler.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_second_round_v0.py
#sh tools/dist_train_DR_4lesions_3001.sh ${config_file} 1
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_second_round_v1.py
#config_file=configs/faster_rcnn_rop/faster_rcnn_x101_64x4d_fpn_1x_ridge_in_one.py
#sh tools/dist_train_DR_4lesions_3001.sh ${config_file} 1
#config_file=configs/faster_rcnn_rop/faster_rcnn_x101_64x4d_fpn_1x_ridge_in_one_with_focal_loss.py
#sh tools/dist_train_DR_4lesions_3001.sh ${config_file} 1

#config_file=configs/faster_rcnn_rop/faster_rcnn_x101_64x4d_fpn_1x_ridge_in_one_with_randflip.py
#config_file=configs/faster_rcnn_rop/faster_rcnn_x101_64x4d_fpn_1x_ridge_in_one_with_randflip_preprocess.py
#config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_with_pseudo_gt_v0.py
config_file=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_crop.py
sh tools/dist_train_DR_4lesions_3001.sh ${config_file} 2