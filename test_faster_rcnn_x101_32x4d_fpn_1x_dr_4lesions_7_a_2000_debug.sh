export CUDA_VISIBLE_DEVICES=1
python3 tools/test.py configs/faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_2000_debug.py /data0/qilei_chen/AI_EYE/BostonAI4DB7_a/work_dirs/faster_rcnn_r50_fpn_1x/epoch_2.pth --gpus 1 --out /data0/qilei_chen/AI_EYE/BostonAI4DB7_a/work_dirs/faster_rcnn_r50_fpn_1x/e1_results.pkl --eval bbox --show