import cv2
import matplotlib.pyplot as plt
import numpy as np

# Khởi tạo mảng
array_size = 64
img = np.array([
    [0,   12,  34,  56,  78,  90,  102, 114],
    [16,  28,  40,  52,  64,  76,  88,  100],
    [32,  44,  56,  68,  80,  92,  104, 116],
    [48,  60,  72,  84,  96,  108, 120, 132],
    [64,  76,  88,  0,   112, 124, 136, 148],
    [80,  92,  104, 116, 128, 140, 152, 164],
    [96,  108, 120, 132, 144, 156, 168, 180],
    [112, 124, 136, 148, 160, 172, 184, 196]
])

# Tìm số lớn nhất
biggest_num = 0
for i in img:
    for j in i:
        if j > biggest_num:
            biggest_num = j

# Tìm L (L này là độ sáng bao phủ cả ma trận)
i, a = 1, 0
while i < biggest_num:
    i = 2 ** a
    a += 1
Lmax = (2 ** (a-1)) - 1

count = {}
ratio = {}
acc_ra={}
tong={}
for i in img:
    for j in i:
        if j not in count:
            count[j] = 1
        else: count[j] += 1

for i in count:
    ratio[i] = count[i] / array_size

# Tính CDF, phần ratio tự xoay theo luôn
sorted_count = sorted(count, reverse=False)
accum_ratio = 0
for i in sorted_count:
    accum_ratio += ratio[i]
    acc_ra[i] = accum_ratio

for i in sorted_count:
    tong_ht = round((Lmax-1)*acc_ra[i])
    tong[i] = tong_ht

#for i in sorted_count:
#    if tong[i] == tong[]:
#        count[i] += count[]
#        acc_ra[i] = count[i] / 64

new_count={}

for i in sorted_count:
    if tong[i] not in new_count:
        new_count[tong[i]] = count[i]
    else:
        new_count[tong[i]] += count[i]
        
new_ratio = {}

for i in new_count:
    new_ratio[i] = new_count[i] / array_size

new_img = np.zeros(img.shape, dtype=np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        new_img[i][j] = tong[img[i][j]]

# Tạo một hình tổng hợp
fig = plt.figure(figsize=(16, 18))
fig.suptitle("CÂN BẰNG HISTOGRAM ẢNH XÁM", fontsize=18)

gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])

# -----------------------------
# 1. Hiển thị ảnh gốc
# -----------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
ax1.set_title("Ảnh gốc")
ax1.axis("off")

# -----------------------------
# 2. Hiển thị ảnh sau cân bằng
# -----------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(new_img, cmap='gray', vmin=0, vmax=255)
ax2.set_title("Ảnh sau cân bằng histogram")
ax2.axis("off")

# -----------------------------
# 3. Histogram ảnh gốc
# -----------------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(count.keys(), count.values())
ax3.set_title("Histogram ảnh gốc")
ax3.set_xlabel("Mức xám")
ax3.set_ylabel("Số pixel")

# -----------------------------
# 4. Histogram ảnh sau cân bằng
# -----------------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(new_count.keys(), new_count.values())
ax4.set_title("Histogram ảnh sau cân bằng")
ax4.set_xlabel("Mức xám")
ax4.set_ylabel("Số pixel")

# -----------------------------
# 5. Bảng ảnh gốc
# -----------------------------
ax5 = fig.add_subplot(gs[2, 0])
ax5.axis("off")

data_old = [
    [i, count[i], f"{ratio[i]:.4f}"]
    for i in sorted_count
]

table_old = ax5.table(
    cellText=data_old,
    colLabels=["Mức xám", "Số lần xuất hiện", "Tần suất"],
    cellLoc="center",
    loc="center"
)

table_old.auto_set_font_size(False)
table_old.set_fontsize(8)
table_old.scale(1, 1.2)

# -----------------------------
# 6. Bảng ảnh sau cân bằng
# -----------------------------
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")


sorted_new_count = sorted(new_count)

data_new = [
    [i, new_count[i], f"{new_ratio[i]:.4f}"]
    for i in sorted_new_count
]

table_new = ax6.table(
    cellText=data_new,
    colLabels=["Mức xám mới", "Số lần xuất hiện", "Tần suất"],
    cellLoc="center",
    loc="center"
)

table_new.auto_set_font_size(False)
table_new.set_fontsize(8)
table_new.scale(1, 1.2)

# Điều chỉnh khoảng cách và hiển thị
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

