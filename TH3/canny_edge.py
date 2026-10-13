import numpy as np
import matplotlib.pyplot as plt
import cv2


def canny_edge_detection(img, threshold1, threshold2):
    image = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(image, threshold1, threshold2)
    return canny


if __name__ == '__main__':
    img = cv2.imread('../images/coins.png', 0)

    canny_img = canny_edge_detection(img, 30, 150)

    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("ảnh gốc")

    ax2.imshow(cv2.cvtColor(canny_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Phát hiện cạnh bằng canny edge detection")

    plt.show()
