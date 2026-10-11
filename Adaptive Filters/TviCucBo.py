import numpy as np
import cv2
import matplotlib.pyplot as plt


def Loc_Thich_Nghi_Cuc_Bo(img, ksize, phuong_sai_nhieu):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros((m, n)).astype(float)
    h = (ksize - 1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i + ksize, j:j + ksize]
            phuong_sai_cuc_bo = np.var(vung_anh_kich_thuoc_k)
            gia_tri_TB_cuc_bo = np.mean(vung_anh_kich_thuoc_k)

            # Nếu phương sai cục bộ nhỏ hơn hoặc bằng phương sai nhiễu, tỷ số giới hạn về 1 (lấy giá trị trung bình)
            if phuong_sai_cuc_bo <= phuong_sai_nhieu:
                img_ket_qua_anh_loc[i, j] = gia_tri_TB_cuc_bo
            else:
                # Công thức: f_hat(x, y) = g(x, y) - (sigma_n^2 / sigma_L^2) * [g(x, y) - mean_L]
                he_so = phuong_sai_nhieu / phuong_sai_cuc_bo
                img_ket_qua_anh_loc[i, j] = padded_img[i + h, j + h] - he_so * (
                            padded_img[i + h, j + h] - gia_tri_TB_cuc_bo)

    return np.uint8(np.clip(img_ket_qua_anh_loc, 0, 255))


if __name__ == "__main__":
    img_nhieu = cv2.imread('../Images/Anh_nhieu_de_loc_thich_nghi_cuc_bo.tif', 0).astype(float)
    ksize = 7
    phuong_sai_nhieu = 1000.0  # Phương sai nhiễu Gaussian sigma_eta^2
    img_ket_qua = Loc_Thich_Nghi_Cuc_Bo(img_nhieu, ksize, phuong_sai_nhieu)

    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.imshow(img_nhieu, cmap='gray')
    ax1.set_title("ảnh nhiễu Gaussian")
    ax1.axis("off")

    ax2.imshow(img_ket_qua, cmap='gray')
    ax2.set_title("ảnh sau khi qua Bộ lọc thích nghi cục bộ")
    ax2.axis("off")
    plt.show()