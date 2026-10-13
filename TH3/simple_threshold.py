import numpy as np
import matplotlib.pyplot as plt
import cv2

def cat_nguong_toan_cuc(image, T , inverse=False):

    #Note: CNTC: T chọn bằng tay 
    # T: Ngưỡng
    if inverse == True:
        (T, thresh) = cv2.threshold(image, T, 255, cv2.THRESH_BINARY_INV)
    else:
        (T, thresh) = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
    return thresh

if __name__ == "__main__":
    img = cv2.imread('images/coins.png',0)
    blurred = cv2.GaussianBlur(img,(5,5),0) # Áp dụng Gaussian blurring với bán kính bằng 5 để loại 1 số cạnh ko quan tâm
    T = 155 # Setup threshold
    anh_cat_nguong = cat_nguong_toan_cuc(img, T)
    anh_cat_nguong_inv = cat_nguong_toan_cuc(img, T, inverse=True)

    # Visual the coin as the mask and seperate the coin with the background

    mask_image = cv2.bitwise_and(img, img,mask=anh_cat_nguong_inv)

    fig = plt.figure(figsize=(12,5))
    (ax1,ax2), (ax3,ax4) =fig.subplots(2,2)

    ax1.imshow(cv2.cvtColor(blurred,cv2.COLOR_BGR2RGB))
    ax1.set_title("Original")

    ax2.imshow(cv2.cvtColor(anh_cat_nguong,cv2.COLOR_BGR2RGB))
    ax2.set_title("Threshold=155/cv2.ThreshBinary")

    ax3.imshow(cv2.cvtColor(anh_cat_nguong,cv2.COLOR_BGR2RGB))
    ax3.set_title("Threshold=155/cv2.ThreshBinaryInv")

    ax4.imshow(cv2.cvtColor(mask_image,cv2.COLOR_BGR2RGB))
    ax4.set_title("Mask/cv2.ThreshBinaryInv")

    plt.show()




