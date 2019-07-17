import os
result = os.popen('nvidia-smi')
lines = result.split('\n')
for line in lines:
    print(line)