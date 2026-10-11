import numpy as np
import cv2
import matplotlib.pyplot as plt


def Loc_Trung_vi_Thich_nghi(img, Smax):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros((m, n)).astype(float)
    # Đệm viền ảnh theo kích thước cửa sổ tối đa Smax để không bao giờ bị vượt biên
    h_max = (Smax - 1) // 2
    padded_img = np.pad(img, (h_max, h_max), mode='reflect')

    for i in range(m):
        for j in range(n):
            # Tọa độ tương ứng trên ảnh đã đệm viền
            x = i + h_max
            y = j + h_max

            # Khởi tạo kích thước cửa sổ lọc ban đầu Sxy = 3
            ksize = 3
            z_xy = padded_img[x, y]

            while True:
                h = (ksize - 1) // 2
                vung_anh = padded_img[x - h: x + h + 1, y - h: y + h + 1]

                z_min = np.min(vung_anh)
                z_max = np.max(vung_anh)
                z_med = np.median(vung_anh)

                # LEVEL A
                A1 = z_med - z_min
                A2 = z_med - z_max
                if A1 > 0 and A2 < 0:
                    # z_min < z_med < z_max -> Chuyển sang LEVEL B
                    B1 = z_xy - z_min
                    B2 = z_xy - z_max
                    if B1 > 0 and B2 < 0:
                        # z_min < z_xy < z_max -> z_xy không phải nhiễu
                        img_ket_qua_anh_loc[i, j] = z_xy
                    else:
                        # z_xy là nhiễu xung -> thay bằng z_med
                        img_ket_qua_anh_loc[i, j] = z_med
                    break
                else:
                    # z_med là nhiễu xung -> Tăng kích thước cửa sổ lọc Sxy
                    ksize += 2
                    if ksize <= Smax:
                        # Lặp lại Level A với cửa sổ lớn hơn
                        continue
                    else:
                        # Vượt quá kích thước Smax cho phép -> xuất z_med
                        img_ket_qua_anh_loc[i, j] = z_med
                        break
    return np.uint8(np.clip(img_ket_qua_anh_loc, 0, 255))

if __name__ == "__main__":
    img_nhieu = cv2.imread('../Images/Anh_nhieu_de_loc_trung_vi_thich_nghi.tif', 0).astype(float)
    Smax = 7
    img_ket_qua_AMF = Loc_Trung_vi_Thich_nghi(img_nhieu, Smax)

    fig = plt.figure(figsize=(12, 6))
    (ax1, ax2) = fig.subplots(1, 2)

    ax1.imshow(img_nhieu, cmap='gray')
    ax1.set_title("Ảnh gốc bị nhiễu muối tiêu mật độ cao")
    ax1.axis("off")

    ax2.imshow(img_ket_qua_AMF, cmap='gray')
    ax2.set_title(f"Ảnh sau lọc Trung vị thích nghi (Smax = {Smax})")
    ax2.axis("off")

    plt.show()