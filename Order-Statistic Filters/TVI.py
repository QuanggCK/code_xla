import numpy as np
import cv2
import matplotlib.pyplot as plt


def Loc_TKTT_Trung_vi(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros((m, n)).astype(float)
    h = (ksize - 1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i + ksize, j:j + ksize]
            # Lấy giá trị trung vị của các điểm ảnh trong vùng cửa sổ lọc
            img_ket_qua_anh_loc[i, j] = np.median(vung_anh_kich_thuoc_k)

    return np.uint8(np.clip(img_ket_qua_anh_loc, 0, 255))


if __name__ == "__main__":
    img_nhieu_muoi_tieu = cv2.imread('../Images/Anh_de_loc_nhieu_thong_ke.tif', 0).astype(float)
    ksize = 3

    # Lọc trung vị lần 1
    img_ket_qua_lan1 = Loc_TKTT_Trung_vi(img_nhieu_muoi_tieu, ksize)
    # Lọc trung vị lần 2 (lấy kết quả lần 1 lọc tiếp)
    img_ket_qua_lan2 = Loc_TKTT_Trung_vi(img_ket_qua_lan1.astype(float), ksize)

    fig = plt.figure(figsize=(16, 9))
    (ax1, ax2, ax3) = fig.subplots(1, 3)

    ax1.imshow(img_nhieu_muoi_tieu, cmap='gray')
    ax1.set_title("Ảnh gốc bị nhiễu muối tiêu")
    ax1.axis("off")

    ax2.imshow(img_ket_qua_lan1, cmap='gray')
    ax2.set_title("Ảnh sau lọc Trung vị lần 1")
    ax2.axis("off")

    ax3.imshow(img_ket_qua_lan2, cmap='gray')
    ax3.set_title("Ảnh sau lọc Trung vị lần 2")
    ax3.axis("off")

    plt.show()