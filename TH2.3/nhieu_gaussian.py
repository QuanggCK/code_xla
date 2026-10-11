import numpy as np
import matplotlib.pyplot as plt
import cv2

# Đọc ảnh
img = cv2.imread('images/anh_nhieu_muoi.tif',0)
m,n = img.shape[:2]

# Thêm nhiễu Gaussian
gtri_TB = 10
psai = 25
noise = np.random.normal(loc=gtri_TB,scale=psai,size=(m,n))
Gau_noisy_img = img + noise

# Thêm nhiễu Rayleigh
psai = 40
noise = np.random.rayleigh(scale=psai, size=(m,n))
Ray_noise_img = img + noise

# Thêm nhiễu Elang 
K = 2.0
psai = 18
noise = np.random.gamma(shape=K, scale=psai, size=(m,n))
Gam_noise_img = noise + img

# Thêm nhiễu mũ
psai = 26
noise = img.random .exponential(scale=psai, size=(m,n))
Exp_noisy_img = noise + img

# Thêm nhiễu mẫu
a,b = 10,100
noise = np.random.uniform(low=a, high=b,size=(m,n))
Uni_noisy_img = noise + img

# Thêm nhiễu muối tiêu (salt & pepper)
number_black = int(m*n*0.05) # Định nghĩa số điểm đen
number_white = int (m*n*0.05) # Định nghĩa số điểm trắng

# Tạo tọa độ ngẫu nhiên cho các điểm nhiễu đen
m_blacks = np.random.randint(0, m, number_black)
n_blacks = np.random.randint(0, n, number_black)

# Tạo tọa độ ngẫu nhiên cho các điểm nhiễu trắng
m_whites = np.random.randint(0, m, number_white)
n_whites = np.random.randint(0, n, number_white)

# Sao chép ảnh gốc
SP_noisy_img = np.copy(img)

# Đặt các pixel được chọn thành màu đen
SP_noisy_img[m_blacks, n_blacks] = 0

# Đặt các pixel được chọn thành màu trắng
SP_noisy_img[m_whites, n_whites] = 255

# 1. Hiển thị ảnh gốc, ảnh nhiễu và histogram
#Tạo 1 cửa sổ hiển thị cho ảnh nhiễu Gaussian
fig1 = plt.figure(figsize=(16,9)) # Vùng vẽ tỉ lệ 16:9

# Tạo 9 vùng vẽ con
(ax1,ax2),(ax3,ax4)=fig1.subplots(2,2)

#Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title("Ảnh gốc")
ax1.axis('off')

# Hiển thị Histogram ảnh gốc
ax2.hist(img.flatten(), bins=256)
ax2.set_title("Histogram")

# Hiển thị ảnh nhiễu Erlang (Gammar)
ax3.imshow(Gam_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu Erlang (Gammar)')
ax3.axis('off')

# Hiển thị histogram ảnh nhiễu Erlang (Gammar)
ax4.hist(Gam_noisy_img.flatten(), bins=256)
ax4.set_title('Histogram')
plt.show()

# Tạo cửa sổ 4 để hiển thị điểm ảnh trên hà số mũ
fig2 = plt.figure(figsize=(16, 9))  # Tạo vùng vẽ tỷ lệ 16:9

# Tạo 4 vùng con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)

# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')

# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

# Hiển thị ảnh nhiễu nhiễu hàm mũ
ax3.imshow(Exp_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu hàm mũ')
ax3.axis('off')

# Hiển thị histogram ảnh nhiễu nhiễu hàm mũ
ax4.hist(Exp_noisy_img.flatten(), bins=256)
ax4.set_title('Histogram')

plt.show()

# 5. Tạo cửa sổ 5 để hiển thị ảnh cho nhiễu nhiễu Uniform
fig2 = plt.figure(figsize=(16, 9))  # Tạo vùng vẽ tỷ lệ 16:9
# Tạo 4 vùng con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)

# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')

# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

# Hiển thị ảnh nhiễu Uniform
ax3.imshow(Uni_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu Uniform')
ax3.axis('off')

# Hiển thị histogram ảnh nhiễu Uniform
ax4.hist(Uni_noisy_img.flatten(), bins=256)
ax4.set_title('Histogram')

plt.show()

# 6. Tạo cửa sổ 6 để hiển thị ảnh cho nhiễu muối tiêu (nhiễu xung)
fig2 = plt.figure(figsize=(16, 9))  # Tạo vùng vẽ tỷ lệ 16:9

# Tạo 4 vùng con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)

# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')

# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

# Hiển thị ảnh nhiễu muối tiêu (nhiễu xung)
ax3.imshow(SP_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu muối tiêu')
ax3.axis('off')

# Hiển thị histogram ảnh nhiễu muối tiêu (nhiễu xung)
ax4.hist(SP_noisy_img.flatten(), bins=256)
ax4.set_title('Histogram')

plt.show()








