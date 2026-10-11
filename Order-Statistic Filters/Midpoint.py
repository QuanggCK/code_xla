import numpy as np
import cv2
import matplotlib.pyplot as plt

def Loc_TKTT_Midpoint(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc_Midpoint = np.zeros([m, n]).astype(float)
    h = (ksize - 1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize, j:j+ksize]
            gia_tri_Max = np.max(vung_anh_kich_thuoc_k)
            gia_tri_Min = np.min(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_Midpoint[i, j] = (gia_tri_Max + gia_tri_Min) / 2
    return np.uint8(np.clip(img_ket_qua_anh_loc_Midpoint, 0, 255))

if __name__ == "__main__":
    img_nhieu_dong_nhat = cv2.imread('../Images/Anh_nhieu_DN_de_loc_MidPoint.tif', 0).astype(float)
    ksize = 5 # Để lọc ảnh Midpoint
    img_KQ_MidPoint = Loc_TKTT_Midpoint(img_nhieu_dong_nhat, ksize)

    fig = plt.figure(figsize=(12, 6))
    (ax1, ax2) = fig.subplots(1, 2)
    ax1.imshow(img_nhieu_dong_nhat, cmap='gray')
    ax1.set_title("ảnh gốc bị nhiễu đồng nhất")
    ax1.axis("off")

    ax2.imshow(img_KQ_MidPoint, cmap='gray')
    ax2.set_title("ảnh sau khi lọc MidPoint")
    ax2.axis("off")
    
    plt.show()