import matplotlib.pyplot as plt
import cv2
import numpy as np

canvas = np.zeros((300, 300, 3), dtype='uint8')

white = (255, 255, 255)

# Tâm 
center = (150, 150)

# Hàm kiểm tra snt
for radius in range(2, 176):
    check = True

    for i in range(2, radius):
        if radius % i == 0:
            check = False
            break

    if check:
        cv2.circle(canvas, center, radius, white, 1)
plt.imshow(canvas)
plt.show()