import matplotlib.pyplot as plt
import cv2
import numpy as np

# Tự tạo 1 hình đen bằng nunmpy
canvas = np.zeros((300,300,3), dtype='uint8') # dtype để lưu các ảnh dạng 8 bit ko dấu

white = (255,255,255)
red = (0,0,255)

# Vẽ các đường
cv2.line(canvas, (0,0), (canvas.shape[0], canvas.shape[1]), white)
cv2.line(canvas, (canvas.shape[0],0), (0, canvas.shape[1]), red)
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

# Vẽ kẻ trục
plt.xlim([0,300-1])
plt.ylim([0,300-1])

plt.show()


