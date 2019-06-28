export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=0

config_dir=configs/faster_rcnn_dr_4lesions/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_baseline.py
model_dir=/data0/qilei_chen/AI_EYE/BostonAI4DB7/work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_baseline

python3 tools/test.py ${config_dir} ${model_dir}/epoch_1.pth --gpus 1 --out ${model_dir}/e1_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_4.pth --gpus 1 --out ${model_dir}/e4_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_5.pth --gpus 1 --out ${model_dir}/e5_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_6.pth --gpus 1 --out ${model_dir}/e6_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_7.pth --gpus 1 --out ${model_dir}/e7_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_8.pth --gpus 1 --out ${model_dir}/e8_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_9.pth --gpus 1 --out ${model_dir}/e9_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_10.pth --gpus 1 --out ${model_dir}/e10_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_11.pth --gpus 1 --out ${model_dir}/e11_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_12.pth --gpus 1 --out ${model_dir}/e12_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_13.pth --gpus 1 --out ${model_dir}/e13_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_14.pth --gpus 1 --out ${model_dir}/e14_results.pkl --eval bbox
#python3 tools/test.py ${config_dir} ${model_dir}/epoch_15.pth --gpus 1 --out ${model_dir}/e15_results.pkl --eval bbox