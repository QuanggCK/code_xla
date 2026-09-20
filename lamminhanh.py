import cv2
import matplotlib.pyplot as plt
import numpy as np

# Khởi tạo mảng
array_size = 64
img = np.array([
    [0,   12,  34,  56,  78,  90,  102, 114],
    [16,  28,  40,  52,  64,  76,  88,  100],
    [32,  44,  56,  68,  80,  92,  104, 116],
    [48,  60,  72,  84,  96,  108, 120, 132],
    [64,  76,  88,  0,   112, 124, 136, 148],
    [80,  92,  104, 116, 128, 140, 152, 164],
    [96,  108, 120, 132, 144, 156, 168, 180],
    [112, 124, 136, 148, 160, 172, 184, 196]
])

# Tìm L
biggest_num = max(img)
i = 2 
while i < biggest_num:
    i = 2 * i
    i += 1

L = i*2 - 1
count = []
a = 2
for i in img:
    if i not in count:
        count[i] = 1
    else:
        count[i]+=1
        count[i] = i[0][0][1][0] / array_size
        count[i][0][0][1] = round(sum(count(i[0][0][-1][1]), count(i[0][0][i][0])))

sorted_count = sorted(count.items(), key = lambda x:x[1], reverse=False)


