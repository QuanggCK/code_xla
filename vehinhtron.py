import matplotlib.pyplot as plt
import cv2
import numpy as np

canvas = np.zeros((300, 300, 3), dtype='uint8')

for i in range(0, 25):
    # Tạo bán kính ngẫu nhiên từ 5 đến 199
    radius = np.random.randint(5, high=200)

    # Tạo màu ngẫu nhiên gồm 3 giá trị B, G, R từ 0 đến 255
    color = np.random.randint(0, high=256, size=(3,)).tolist()

    # Tạo tọa độ tâm hình tròn ngẫu nhiên (x, y)
    # Mỗi tọa độ nằm trong khoảng từ 0 đến 299
    pt = np.random.randint(0, high=300, size=(2,))
    
    # Vẽ hình tròn lên canvas
    # tuple(pt): chuyển tọa độ thành tuple (x, y)
    #radius: bán kính hình tròn
    # color: màu sắc hình tròn
    # -1: tô kín hình tròn
    cv2.circle(canvas, tuple(pt), radius, color, -1)

plt.imshow(canvas)
plt.show()  