import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


img = plt.imread('images/trex.png')
fig = plt.figure()
ax1,ax2 = fig.subplots(1,2)

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original")

slice= img[30:120,240:335]
ax2.imshow(cv2.cvtColor(slice, cv2.COLOR_BGR2RGB))
ax2.set_title("Cropped")
ax2.set_xlim(0,90)
ax2.set_ylim(95,0)

plt.savefig("crop.pdf", bbox_inches='tight')
plt.show()

