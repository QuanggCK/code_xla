import numpy as np
import matplotlib.pyplot as plt
import cv2


def image_translate(image, shiftX, shiftY):
    """
    Translate (shift) an image using a NumPy matrix.
    shiftX: số pixel dịch chuyển theo trục X.
    shiftY: số pixel dịch chuyển theo trục Y.
    """

    M = np.float32([
        [1, 0, shiftX],
        [0, 1, shiftY]
    ])

    return cv2.warpAffine(
        image,M,(image.shape[1], image.shape[0])
    )


img = cv2.imread('./images/trex.png')

fig = plt.figure()
ax1, ax2 = fig.subplots(1, 2)

# Hiển thị ảnh gốc
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original")

# Dịch chuyển ảnh theo trục X và Y
shifted = image_translate(img, 10, 20)

ax2.imshow(cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB))
ax2.set_title("Translated image")
h, w = img.shape[:2]

ax2.set_xlim(10, w)
ax2.set_ylim(h, 20)

plt.savefig("translated.pdf", bbox_inches='tight')
plt.show()