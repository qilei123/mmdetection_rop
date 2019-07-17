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
        #print(infos)
        #print(infos[17])
        info_index = 0
        for info in infos:
            if 'MiB' in info:
                break
            info_index+=1
        if 'MiB' in infos[int(info_index)]:
            memory_use = int(infos[int(info_index)][:-3])
            print('gpu_id:'+str(gpu_id)+' | used memory:'+infos[int(info_index)])
        #print(infos)
        if memory_use<memory_limit:
            break

single_gpu_check_and_wait(1,0)

command = 'sh test_faster_rcnn_x101_32x4d_fpn_1x_dr_4lesions_7_a_with_focal_loss_smallset_advance_optdataset4_deephead_v1_InstanceBalancedPosSampler.sh'

#os.system(command)