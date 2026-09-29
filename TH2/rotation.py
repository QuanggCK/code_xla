# Rotate 1 image 10 degree around pixel(0,0)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Hàm xoay ảnh
def image_rotation(image, degree):
    theta = degree * math.pi / 180  # chuyển đổi sang radian
    (h, w) = image.shape[:2]
    M = np.float32([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0]
    ])
    return cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
)

img = cv2.imread('images/trex.png')
fig = plt.figure()
ax1,ax2 = fig.subplots(1,2)

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Orginal")

rotated = image_rotation(img,10)

ax2.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
ax2.set_title("Rotated")

plt.savefig("translated_pic.pdf", bbox_inches='tight')
plt.show()