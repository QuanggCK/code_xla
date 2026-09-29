import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def image_rotation(image, degree):
    theta = degree * math.pi / 180
    (h, w) = image.shape[:2]

    MT1 = np.float32([
        [1, 0, w/2],
        [0, 1, h/2]
    ])

    MR = np.float32([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0, 0, 1]
    ])

    MT2 = np.float32([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0, 1]
    ])

    # M: final transformation matrix
    M = MT1 @ MR @ MT2

    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255,255,255))

img = cv2.imread('images/trex.png')
fig = plt.figure()
ax1,ax2 = fig.subplots(1,2)
ax1.imshow(img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Orginal")

rotated = image_rotation(img)
ax2.imshow(rotated, cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
ax2.set_title("Rotated Center")

plt.savefig("rotated_cen.pdf", bbox='tight')

plt.show()