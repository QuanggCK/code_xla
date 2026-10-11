import numpy as np
import cv2 
import matplotlib.pyplot as plt

def Loc_TB_SoHoc (img, ksize):
    m,n = img.shape
    img_kqua_anh_loc = np.zeros([m,n])
    h = (ksize -1) / 2 # Tính thông số h để padding bức ảnh phù hợp với kích thước ksize của bộ lọc
    padded_img = np.pad(img, (h,h), mode = 'reflect') # Mở rộng 4 biên của bức ảnh theo thông số h
    for i in range (m):
        for j in range (n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize, j:j+ksize]
            img_kqua_anh_loc[i,j] = np.mean(vung_anh_kich_thuoc_k) 
            # Phương thức hỗ trợ tính trung bình cường độ sáng trong vùng S_{x,y}
    return img_kqua_anh_loc

# Hàm main
if __name__ == __main__:
    img_nhieu= cv2.imread('images/anh_nhieu_de_loc_rung_binh.tif',0)
    ksize = 5
    img_kqua_tbsh = Loc_TB_SoHoc(img_nhieu, ksize)

    fig = plt.figure(figsize=(16,9))
    ax1,ax2 = fig.subplots(1,2)
    ax1.imshow(img_nhieu, cmap='gray')
    ax1. set_title("Ảnh nhiễu")

    ax2.imshow(img_kqua_tbsh, cmap='gray')
    ax2.set_title("Ảnh lọc sau nhiễu")

    plt.show()