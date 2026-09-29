import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def image_flipping(image,flip_model=0):
    return cv2.flip(image,flipCode=flip_model)

if __name__ == "__main__":
    img = cv2.imread('images/trex.png')
    fig = plt.figure()
    ax1,ax2 = fig.subplots(1,2)

    ax1.set_title("Original")
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax2.set_title("Flipped")
    flipped_image = image_flipping(img, flip_model=0)
    # 0: Lật ngược lại (hồ nước)
    # 1: Lật trái
    # -1: Lật dọc ngang (bắt chước tập thể dục)
    ax2.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))

    plt.savefig("flip.pdf")
    plt.show()
