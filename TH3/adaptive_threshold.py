import numpy as np
import matplotlib.pyplot as plt
import cv2

def cat_nguong_thich_nghi(image, neighborhood='mean', ksize=3, int_c=4):
    """
    Hàm cắt ngưỡng thích nghi, tự động xác định ngưỡng T cho từng vùng ảnh.

    image: Ảnh đầu vào (ảnh xám).
    neighborhood: Phương pháp tính ngưỡng ('mean' hoặc 'gaussian').
    ksize: Kích thước cửa sổ lân cận, phải là số lẻ lớn hơn 1.
    int_c: Hằng số C được trừ khỏi ngưỡng cục bộ.
    """
    # Cắt ngưỡng thích nghi Mean
    if neighborhood == 'mean':
        thresh = cv2.adaptiveThreshold(image,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            ksize,
            int_c
        )
    # Cắt ngưỡng thích nghi Gaussian
    elif neighborhood == 'gaussian':
        thresh = cv2.adaptiveThreshold(image,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ksize,
            int_c
        )

    # Nếu phương pháp không hợp lệ, trả về ảnh ban đầu
    else:
        return image
    return thresh

if __name__ == "__main__":
    img = cv2.imread('./images/coins.png', 0)
    if img is None:
        print("Không đọc được ảnh! Hãy kiểm tra đường dẫn.")
        exit()

    # Làm mờ Gaussian để giảm nhiễu
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Cắt ngưỡng thích nghi Mean
    anh_cat_nguong_mean = cat_nguong_thich_nghi(
        blurred, neighborhood='mean', ksize=11, int_c=4
    )

    # Cắt ngưỡng thích nghi Gaussian
    anh_cat_nguong_gauss = cat_nguong_thich_nghi(
        blurred, neighborhood='gaussian', ksize=15, int_c=3
    )

    # Hiển thị kết quả
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2, ax3 = fig.subplots(1, 3)

    ax1.imshow(img, cmap='gray')
    ax1.set_title("Ảnh gốc")

    ax2.imshow(anh_cat_nguong_mean, cmap='gray')
    ax2.set_title("Cắt ngưỡng thích nghi Mean")

    ax3.imshow(anh_cat_nguong_gauss, cmap='gray')
    ax3.set_title("Cắt ngưỡng thích nghi Gaussian")

    for ax in (ax1, ax2, ax3):
        ax.axis('off')

    plt.show()
