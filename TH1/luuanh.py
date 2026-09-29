from __future__ import print_function
import argparse
import cv2

# Khởi tạo bộ xử lý tham số (Construct Parser) và xử lý tham số
ap = argparse.ArgumentParser()
#ap.add_argument("-i","--images", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# Chạy ảnh và hiện thông tin đơn giản
image = cv2.imread(args["images"])
print("Width: {} pixels.".format(image.shape[1]))
print("Height: {} pixels.".format(image.shape[0]))
print("Channels: {}".format(image.shape[2]))

# Hiện ảnh và tạo nút tắt
cv2.imshow("Image",image)
cv2.waitKey(0) # Nhấn phím bất kì, tgian chờ không có (0)

# Lưu ảnh, OpenCV có thể xử lý việc chuyển đổi loại file 1 cách tự động
cv2.imwrite("newimage.jpg", image)



