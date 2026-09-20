import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    image = cv2.imread("images/keodan_dau.tif",cv2.IMREAD_GRAYSCALE) # Đọc ảnh bằng OpenCV và đọc ảnh dạng mức xám
    plt.figure(figsize=(12 , 4))
    plt.imshow(image, cmap="gray",vmin=0, vmax=255) # Ảnh gốc
    plt.title("Original Image")
    plt.show()
    # Ảnh gốc