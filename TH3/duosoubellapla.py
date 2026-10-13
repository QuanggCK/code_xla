import numpy as np
import matplotlib.pyplot as plt
import cv2

def sobel_filter(image):
    sobelX = cv2.Sobel (image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel (image, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute (sobelX))
    sobely = np.uint8(np.absolute (sobely))

    sobelCombined = cv2.bitwise_or (sobelX, sobely)
    return sobelCombined

def laplacian_filter(image):
    lap = cv2.Laplacian (image, cv2.CV_64F)
    lap = np.uint8(np.absolute (lap))
    return lap

if __name__ == '__main__':
    img = cv2.imread('../images/coins.png', 0)
    sobel_img = sobel_filter(img)
    lap_img = laplacian_filter(img)

    fig = plt.figure(figsize=(16, 9))
    ax1, ax2, ax3 = fig.subplots (1, 3)

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("ảnh gốc")

    ax2.imshow(cv2.cvtColor(sobel_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Phát hiện cạnh bằng đạo hàm sobel")

    ax3.imshow(cv2.cvtColor(lap_img, cv2.COLOR_BGR2RGB))
    ax3.set_title("Phát hiện cạnh bằng đạo hàm laplacian")

    plt.show()
