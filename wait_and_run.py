import os
import time

def single_gpu_check_and_wait(gpu_id,memory_limit):
    count=0
    while True:
        count+=1
        if count==100:
            os.system('clear')
            count=0
        memory_use = 100000
        time.sleep(1)
        result = os.popen('nvidia-smi').read()
        #print(result)
        lines = result.split('\n')
        gpu_info_line = lines[8+gpu_id*3]
        #print(gpu_info_line)
        infos = gpu_info_line.split(' ')
        if 'MiB' in infos[17]:
            memory_use = int(infos[17][:-3])
            print(infos[17])
        #print(infos)
        if memory_use<memory_limit:
            break

single_gpu_check_and_wait(1,5000)

command = 'sh test_faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_IoUBalancedNegSampler.sh'

os.system(command)