git pull
export CUDA_VISIBLE_DEVICES=0
img_dir=/data0/qilei_chen/old_alien/AI_EYE_IMGS/ROP_DATASET_with_label/2TISSUES/
config_dir=configs/mask/mask_rcnn_x101_64x4d_fpn_1x_2tissues.py
model_dir=/data0/qilei_chen/old_alien/AI_EYE_IMGS/ROP_DATASET_with_label/2TISSUES/mask_epoch_12.pth
python3 inference_single_img.py --img_dir ${img_dir} --config_dir ${config_dir} --model_dir ${model_dir}
