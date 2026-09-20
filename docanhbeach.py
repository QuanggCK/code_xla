import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/beach.png')

# In thông số ánh
print(f"Width of image: {img.shape[1]} pixels") # Chiều dài (ngang)
print(f"Heights of image: {img.shape[0]} pixels") # Chiều cao (hàng)
print(f"Channels of image: {img.shape[2]}")

#Vẽ ảnh bằng matplotib
fig = plt.figure()

# Đổi màu ngược lại theo cơ chế RGB
#plt.imshow(img) #B=0,G=1, R=2
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #B=0,G=1, R=2
plt.show()
cv2.imwrite("trieu_newfigure.png", img)



