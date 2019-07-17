import os
result = (os.system('nvidia-smi'))
lines = result.split('\n')
for line in lines:
    print(line)