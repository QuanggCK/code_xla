# import thu vien
import matplotlib.pyplot as plt
import cv2
import numpy as np


def Convolution2D(img, kernel):
    '''
    Code đang không mở rộng biên
    '''
    m, n = img.shape

    img_new = np.zeros([m, n])

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            img_new[i, j] = np.sum(
                img[i - 1:i + 2, j - 1:j + 2] * kernel
            )

    img_new = img_new.astype(np.uint8)

    return img_new


def laplacian_filter(img):
    '''Bộ lọc Laplacian chuẩn'''

    Laplacian_kerner = np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ],
        dtype=np.float32
    )

    return Convolution2D(img, Laplacian_kerner)


if __name__ == '__main__':

    # Đọc và hiển thị ảnh gốc
    image = cv2.imread("./figures/laplapcian.tif", 0)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots(2, 2)

    # Hiển thị ảnh gốc
    ax[0, 0].set_title("Original")
    ax[0, 0].imshow(image, cmap="gray")

    # Hiển thị ảnh lọc Laplacian
    laplacian_image = laplacian_filter(image)

    ax[0, 1].set_title("laplacian_filter")
    ax[0, 1].imshow(laplacian_image, cmap="gray")

    # Hiển thị ảnh sau khi lọc
    ax[1, 0].set_title("Anh sau khi loc laplacian")
    ax[1, 0].imshow(
        image - laplacian_image,
        cmap="gray"
    )

    # Tắt ô không sử dụng
    ax[1, 1].axis('off')

    plt.show()