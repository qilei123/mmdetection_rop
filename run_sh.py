import os
import time

while True:
    time.sleep(1)
    result = os.popen('nvidia-smi').read()
    print(result)
    lines = result.split('\n')
    gpu_info_line = lines[8]
    print(gpu_info_line)
    infos = gpu_info_line.split(' ')
    print(infos)