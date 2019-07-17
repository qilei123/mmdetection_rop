import os
import time

while True:
    os.system('clear')
    time.sleep(1)
    result = os.popen('nvidia-smi').read()
    print(result)
    lines = result.split('\n')
    gpu_info_line = lines[8]
    print(gpu_info_line)
    infos = gpu_info_line.split(' ')
    print(int(infos[17][:-3]))
    print(infos)
    