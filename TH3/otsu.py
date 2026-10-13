import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas

def otsu_threshold(image):
    T = mahotas.thresholding.otsu(image)
    print("Otsu's threshold: {}".format(T))
    thresh = image.copy()
    thresh[thresh > T] = 255
    thresh[thresh < 255] = 0
    thresh = cv2.bitwise_not(
        thresh)
    return thresh


if __name__ == '__main__':
    img = cv2.imread('../images/coins.png', 0)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    anh_cat_nguong_otsu = otsu_threshold(blurred)

    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax1.set_title("ảnh gốc")

    ax2.imshow(cv2.cvtColor(anh_cat_nguong_otsu, cv2.COLOR_BGR2RGB))
    ax2.set_title("ảnh cắt ngưỡng toàn cục, tìm T bằng otsu")

    plt.show()
