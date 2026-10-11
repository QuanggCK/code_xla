import numpy as np
import cv2
import matplotlib.pyplot as plt

def Loc_Trung_binh_cat_Alpha(img, ksize, alpha):
    m, n = img.shape
    img_LQ_Trung_binh_cat_Alpha = np.zeros([m, n]).astype(float)
    h = (ksize - 1) // 2
    d = int(ksize * ksize * alpha)
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize, j:j+ksize].flatten()
            vung_anh_kich_thuoc_k.sort()
            vung_anh_kich_thuoc_con_lai = vung_anh_kich_thuoc_k[d//2:-d//2]
            img_LQ_Trung_binh_cat_Alpha[i, j] = np.sum(vung_anh_kich_thuoc_con_lai) / (ksize**2 - d)
    return np.uint8(img_LQ_Trung_binh_cat_Alpha)

if __name__ == "__main__":
    img_nhieu_dong_nhat = cv2.imread('../Images/Anh_nhieu_DN_de_loc_MidPoint.tif', 0).astype(float)
    ksize = 5
    alpha = 0.25
    img_KQ_TBCA = Loc_Trung_binh_cat_Alpha(img_nhieu_dong_nhat, ksize, alpha)

    fig = plt.figure(figsize=(12, 6))
    (ax1, ax2) = fig.subplots(1, 2)
    ax1.imshow(img_nhieu_dong_nhat, cmap='gray')
    ax1.set_title("ảnh gốc bị nhiễu")
    ax1.axis("off")

    ax2.imshow(img_KQ_TBCA, cmap='gray')
    ax2.set_title("ảnh sau khi lọc cắt TB Alpha")
    ax2.axis("off")
    plt.show()