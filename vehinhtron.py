import matplotlib.pyplot as plt
# %matplotlib inline
import cv2
import numpy as np

canvas = np.zeros((300, 300, 3), dtype='uint8')

for i in range(0, 25):
    radius = np.random.randint(5, high=200)
    color = np.random.randint(0, high=256, size=(3,)).tolist()
    pt = np.random.randint(0, high=300, size=(2,))
    cv2.circle(canvas, tuple(pt), radius, color, -1)

plt.imshow(canvas)
plt.show()