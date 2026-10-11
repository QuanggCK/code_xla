import numpy as np
import cv2
import matplotlib.pyplot as plt


def Loc_Trung_binh_hinh_hoc(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros((m, n)).astype(float)
    h = (ksize - 1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i + ksize, j:j + ksize]
            # Tính tích các điểm ảnh rồi lấy căn bậc (ksize^2)
            # Dùng exp(mean(log)) để tránh tràn số khi nhân tích các pixel
            img_ket_qua_anh_loc[i, j] = np.exp(np.mean(np.log(vung_anh_kich_thuoc_k.astype(float) + 1e-5)))

    return np.uint8(np.clip(img_ket_qua_anh_loc, 0, 255))


if __name__ == "__main__":
    img_nhieu = cv2.imread('../Images/Anh_nhieu_de_loc_Trung_binh.tif', 0).astype(float)
    ksize = 5
    img_ket_qua_TBHH = Loc_Trung_binh_hinh_hoc(img_nhieu, ksize)

    fig = plt.figure(figsize=(16, 9))
    (ax1, ax2) = fig.subplots(1, 2)
    ax1.imshow(img_nhieu, cmap='gray')
    ax1.set_title("ảnh gốc bị nhiễu")
    ax1.axis("off")

    ax2.imshow(img_ket_qua_TBHH, cmap='gray')
    ax2.set_title("ảnh sau khi lọc Trung bình hình học")
    ax2.axis("off")
    plt.show()