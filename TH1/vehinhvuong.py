import matplotlib.pyplot as plt
import cv2
import numpy as np

plt.rcParams.update({'text.usetex': True})

canvas = np.zeros((300, 300, 3), dtype='uint8')

white = (255, 255, 255)
red = (255, 0, 0)

cv2.rectangle(canvas, (10, 10), (canvas.shape[0] // 4, canvas.shape[1] // 4), white, 2)

cv2.rectangle(canvas, (100, 100), (200, 200), red, -1)

plt.imshow(canvas)
plt.show()