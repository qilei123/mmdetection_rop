import os
result = os.popen('nvidia-smi')
print(result)
lines = result.split('\n')
for line in lines:
    print(line)