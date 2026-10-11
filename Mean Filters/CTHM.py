import numpy as np
import cv2
import matplotlib.pyplot as plt


def Loc_Trung_binh_Contraharmonic(img, ksize, Q):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros((m, n)).astype(float)
    h = (ksize - 1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i + ksize, j:j + ksize]
            # Tính tử số: tổng các giá trị lũy thừa bậc Q + 1
            tu_so = np.sum(vung_anh_kich_thuoc_k ** (Q + 1))
            # Tính mẫu số: tổng các giá trị lũy thừa bậc Q
            mau_so = np.sum(vung_anh_kich_thuoc_k ** Q)

            if mau_so != 0:
                img_ket_qua_anh_loc[i, j] = tu_so / mau_so
            else:
                img_ket_qua_anh_loc[i, j] = 0
    return np.uint8(np.clip(img_ket_qua_anh_loc, 0, 255))

if __name__ == "__main__":
    img_nhieu_muoi = cv2.imread('../Images/Anh_nhieu_muoi.tif', 0).astype(float)
    img_nhieu_tieu = cv2.imread('../Images/Anh_nhieu_hat_tieu.tif', 0).astype(float)
    ksize = 3

    # Với nhiễu hạt tiêu: chọn Q > 0 (ví dụ Q = 1.5)
    img_ket_qua_tieu = Loc_Trung_binh_Contraharmonic(img_nhieu_tieu, ksize, Q=1.5)

    # Với nhiễu muối: chọn Q < 0 (ví dụ Q = -1.5)
    img_ket_qua_muoi = Loc_Trung_binh_Contraharmonic(img_nhieu_muoi, ksize, Q=-1.5)

    fig = plt.figure(figsize=(9, 9))
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

    ax1.imshow(img_nhieu_tieu, cmap='gray')
    ax1.set_title("ảnh gốc bị nhiễu hạt tiêu")
    ax1.axis("off")

    ax2.imshow(img_ket_qua_tieu, cmap='gray')
    ax2.set_title("lọc Contraharmonic (Q = 1.5)")
    ax2.axis("off")

    ax3.imshow(img_nhieu_muoi, cmap='gray')
    ax3.set_title("ảnh gốc bị nhiễu muối")
    ax3.axis("off")

    ax4.imshow(img_ket_qua_muoi, cmap='gray')
    ax4.set_title("lọc Contraharmonic (Q = -1.5)")
    ax4.axis("off")

    plt.show()