import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/trex.png')
fig = plt.figure()

# axes1,2 là chia 2 vùng hiển thị con trên 1 vùng
ax1, ax2 = fig.subplots(1,2)
#  subplots(số hàng, số cột)

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("original")
img[0:50,0:50,:] = (0,0,255) # Chuyển nguyên 50x50 ở đầu thành màu đỏ

ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax2.set_title("updated")
plt.show()