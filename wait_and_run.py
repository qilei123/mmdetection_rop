import os
import time

def single_gpu_check_and_wait(gpu_id,memory_limit):
    while True:
        #os.system('clear')
        time.sleep(1)
        result = os.popen('nvidia-smi').read()
        #print(result)
        lines = result.split('\n')
        gpu_info_line = lines[8+gpu_id*3]
        #print(gpu_info_line)
        infos = gpu_info_line.split(' ')
        memory_use = int(infos[17][:-3])
        print(int(infos[17][:-3]))
        #print(infos)
        if memory_use<memory_limit:
            break

single_gpu_check_and_wait(0,9000)

command = 'sh test_faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_InstanceBalancedPosSampler.sh'

os.system(command)