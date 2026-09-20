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

# Tìm số lớn nhất
biggest_num = 0
for i in img:
    for j in i:
        if j > biggest_num:
            biggest_num = j

# Tìm L (L này là độ sáng bao phủ cả ma trận)
i = 1, a = 0
while i < biggest_num:
    i = 2 ** a
    a += 1
Lmax = (2 ** (a-1)) - 1

count = {}
ratio = {}
acc_ra={}
for i in img:
    for j in i:
        if j not in count:
            count[j] = 1
        else: count[j] += 1

for i in count:
    ratio[i] = count[i] / array_size

# Tính CDF, phần ratio tự xoay theo luôn
sorted_count = sorted(count, reverse=False)
accum_ratio = 0
for i in sorted_count:
    accum_ratio += ratio[i]
    acc_ra[i] = accum_ratio
