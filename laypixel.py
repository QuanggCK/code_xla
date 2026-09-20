import matplotlib.pyplot as plt
import cv2

img = cv2.imread('images/trex.png')

img2= img[0:100,0:100,:]  # Lấy 100x100 pixels ở góc trên cùng

# Lấy pixel góc trên cùng bên trái theo thứ tự  B, G, R
(b_tl,g_tl,r_tl) = img[0,0,:]

# Lấy pixel góc dưới cùng bên phải theo thứ tự B, G, R
(b_br,g_br,r_br) = img[-1,-1,:] # -1 là phần tử cuối cùng

print("Pixels at (0,0): Red - {r_tl} | Blue - {b_tl} | Green - {g_tl}")
print("Pixels at ({img.shape[0]-1},{img.shape[1]-1}: Red - {r_br} | Blue - {b_br} | Green - {g_br}")



plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
cv2.imwrite("crop100pixel.png", img2)
plt.show()