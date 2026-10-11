import numpy as np
import cv2
import matplotlib.pyplot as plt


def Loc_TKTT_Min(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros((m, n)).astype(float)
    h = (ksize - 1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i + ksize, j:j + ksize]
            # Lấy giá trị nhỏ nhất trong vùng cửa sổ lọc
            img_ket_qua_anh_loc[i, j] = np.min(vung_anh_kich_thuoc_k)

    return np.uint8(np.clip(img_ket_qua_anh_loc, 0, 255))


if __name__ == "__main__":
    img_nhieu_muoi = cv2.imread('../Images/Anh_nhieu_muoi.tif', 0).astype(float)
    ksize = 3
    img_ket_qua_Min = Loc_TKTT_Min(img_nhieu_muoi, ksize)

    fig = plt.figure(figsize=(12, 6))
    (ax1, ax2) = fig.subplots(1, 2)

    ax1.imshow(img_nhieu_muoi, cmap='gray')
    ax1.set_title("Ảnh gốc bị nhiễu muối")
    ax1.axis("off")

    ax2.imshow(img_ket_qua_Min, cmap='gray')
    ax2.set_title("Ảnh sau khi lọc Min (ksize = 3)")
    ax2.axis("off")

    plt.show()