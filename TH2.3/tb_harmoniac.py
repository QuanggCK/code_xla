import numpy as np
import cv2 
import matplotlib.pyplot as plt

def Loc_TB_Harmoniac (img, ksize):
    m,n = img.shape
    img_kqua_loc = np.zeros([m,n]).astype(float)
    h = (ksize-1)/2
    padded_img = np.pad(img,(h,h), mode='reflect')
    for i in range (m):
        for j in range (n):
            img_kqua_loc[i,j] = ksize ** 2/np.sum(1.0/padded_img[i:i+1,j:j+1])
    return np.uint8(img_kqua_loc)

if __name__ == __main__:
    img_nhieu_muoi = cv2.imread('images/Anh_nhieu_muoi.tif',0).astype(float)
    img_nhieu_tieu = cv2.imread('images/Anh_nhieu_hat_tieu.tif',0).astype(float)
    ksize=5
    img_kqua_TbHarmoniacMuoi = Loc_TB_Harmoniac(img_nhieu_muoi,kisze)
    img_kqua_TbHarmoniacTieu = Loc_TB_Harmoniac(img_nhieu_hat_tieu,kisze)
    fig = plt.figure(figsize=(9, 9))

(ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

ax1.imshow(img_nhieu_muoi, cmap='gray')
ax1.set_title('ảnh gốc bị nhiễu muối')
ax1.axis("off")

ax2.imshow(img_ket_qua_TBHarmonic_muoi, cmap='gray')
ax2.set_title("ảnh sau khi lọc Trung bình Harmonic")
ax2.axis("off")

ax3.imshow(img_nhieu_tieu, cmap='gray')
ax3.set_title('ảnh gốc bị nhiễu hạt tiêu')
ax3.axis("off")

ax4.imshow(img_ket_qua_TBHarmonic_tieu, cmap='gray')
ax4.set_title("ảnh sau khi lọc Trung bình Harmonic")
ax4.axis("off")

plt.show()
