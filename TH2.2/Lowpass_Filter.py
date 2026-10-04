import matplotlib.pyplot as plt
import cv2
import numpy as np


class lowpass_filter(object):
    """
    Lớp triển khai các thuật toán Lọc tần số thấp (Low-pass Filter) và Lọc phi tuyến (Rank-order Filter)
    để làm mịn ảnh, giảm nhiễu. Tất cả các phép toán tích chập và xử lý viền đều được lập trình
    thủ công (vòng lặp) giúp hiểu rõ bản chất xử lý ảnh mức thấp.
    """

    def __init__(self, image, convolution=None) -> None:
        """
        Khởi tạo đối tượng bộ lọc.
        :param image: Mảng 2D biểu diễn ảnh xám (ma trận các điểm ảnh từ 0 đến 255)
        :param convolution: Cờ (True/False) xác định có khởi tạo các ma trận Kernel hay không
        """
        self.img = image
        self.convolution = convolution

        # Nếu truyền convolution=True, gọi hàm tạo sẵn các mặt nạ lọc (Kernels)
        if convolution:
            self.init_kernel()

    def init_kernel(self):
        """
        Khởi tạo các ma trận trọng số (Kernel/Mặt nạ) kích thước 3x3.
        Lưu ý: Tổng các phần tử trong Kernel lọc tần số thấp thường bằng 1
        để bảo toàn độ sáng trung bình của bức ảnh gốc sau khi lọc.
        """
        # 1. BỘ LỌC TRUNG BÌNH KÍCH THƯỚC 3x3 (Mean Filter):
        # Tất cả 9 phần tử đều có trọng số bằng nhau (1/9).
        # Điểm ảnh mới = (Tổng 9 điểm ảnh xung quanh) / 9.
        self.MeanKernel33 = np.array([[1 / 9, 1 / 9, 1 / 9],
                                      [1 / 9, 1 / 9, 1 / 9],
                                      [1 / 9, 1 / 9, 1 / 9]], dtype=float)

        # 2. BỘ LỌC TRUNG BÌNH CÓ TRỌNG SỐ (Weighted Mean Filter):
        # Pixel trung tâm có trọng số cao nhất (4/16), các pixel lân cận có trọng số giảm dần.
        # Giúp làm mịn ảnh nhưng ít làm mờ các chi tiết quan trọng hơn bộ lọc Mean thông thường.
        # Tổng các phần tử: 1/16 + 2/16 + 1/16 + 2/16 + 4/16 + 2/16 + 1/16 + 2/16 + 1/16 = 16/16 = 1.
        self.MeanWeightKernel33 = np.array([[1 / 16, 2 / 16, 1 / 16],
                                            [2 / 16, 4 / 16, 2 / 16],
                                            [1 / 16, 2 / 16, 1 / 16]], dtype=float)

        # 3. BỘ LỌC GAUSSIAN 3x3:
        # Trọng số được tính theo hàm phân bố chuẩn Gauss (hình chuông), tâm cao nhất và giảm dần ra viền.
        gauss = np.array([[0.0751, 0.1238, 0.0751],
                          [0.1238, 0.2042, 0.1238],
                          [0.0751, 0.1238, 0.0751]], dtype=float)
        # Chia cho tổng (np.sum) để đảm bảo chuẩn hóa tổng trọng số đúng bằng 1
        self.GaussianKernel33 = gauss / np.sum(gauss)

    def _apply_padding(self, border_mode='replicate', pad_width=1):
        """
        Hàm xử lý mở rộng biên ảnh (Padding):
        Khi trượt cửa sổ 3x3 ở rìa ảnh, các ô nằm ngoài biên sẽ không có giá trị.
        Hàm này tạo một vùng đệm rộng 'pad_width' pixel xung quanh ảnh để tránh lỗi vượt dải chỉ số.

        :param border_mode: Chế độ điền biên ('zero', 'replicate', hoặc 'reflect')
        :param pad_width: Độ rộng viền mở rộng (mặc định = 1 cho cửa sổ 3x3)
        :return: Mảng 2D ảnh đã được cộng viền đệm
        """
        m, n = self.img.shape  # Lấy kích thước ảnh gốc: m (chiều cao), n (chiều rộng)

        # Tính kích thước ảnh mới sau khi cộng thêm viền ở cả 2 phía (trên-dưới, trái-phải)
        padded_m = m + 2 * pad_width
        padded_n = n + 2 * pad_width

        # Khởi tạo khung ảnh đệm với toàn giá trị 0 (dạng số thực float)
        padded_img = np.zeros((padded_m, padded_n), dtype=float)

        # Duyệt qua từng tọa độ (i, j) trên khung ảnh đệm mới
        for i in range(padded_m):
            for j in range(padded_n):
                # Quy đổi tọa độ (i, j) của ảnh đệm về tọa độ tương ứng (r, c) trên ảnh gốc:
                # Vì viền bị lệch pad_width pixel nên: r = i - pad_width, c = j - pad_width
                r = i - pad_width
                c = j - pad_width

                # CHẾ ĐỘ 1: 'zero' - Gán giá trị 0 cho tất cả điểm ảnh đệm nằm ngoài biên gốc
                if border_mode == 'zero':
                    if 0 <= r < m and 0 <= c < n:
                        padded_img[i, j] = self.img[r, c]  # Điểm nằm trong ảnh gốc
                    else:
                        padded_img[i, j] = 0  # Điểm nằm ngoài biên

                # CHẾ ĐỘ 2: 'replicate' - Nhân bản điểm ảnh sát viền gần nhất
                elif border_mode == 'replicate':
                    # Ép chỉ số dòng r về phạm vi [0, m-1]
                    if r < 0:
                        r_idx = 0  # Nếu vượt biên trên -> Lấy dòng đầu tiên (0)
                    elif r >= m:
                        r_idx = m - 1  # Nếu vượt biên dưới -> Lấy dòng cuối cùng (m-1)
                    else:
                        r_idx = r  # Nằm trong biên -> Giữ nguyên

                    # Ép chỉ số cột c về phạm vi [0, n-1]
                    if c < 0:
                        c_idx = 0  # Nếu vượt biên trái -> Lấy cột đầu tiên (0)
                    elif c >= n:
                        c_idx = n - 1  # Nếu vượt biên phải -> Lấy cột cuối cùng (n-1)
                    else:
                        c_idx = c  # Nằm trong biên -> Giữ nguyên

                    padded_img[i, j] = self.img[r_idx, c_idx]

                # CHẾ ĐỘ 3: 'reflect' - Lấy đối xứng điểm ảnh qua đường biên
                elif border_mode == 'reflect':
                    # Tính toán chỉ số dòng r bằng phép lấy đối xứng:
                    if r < 0:
                        r_idx = -r  # Ví dụ: r = -1 -> Lấy dòng 1
                    elif r >= m:
                        r_idx = 2 * m - 2 - r  # Ví dụ: m=100, r=100 -> r_idx = 198-100 = 98 (lùi lại 1 bước)
                    else:
                        r_idx = r

                    # Tính toán chỉ số cột c bằng phép lấy đối xứng:
                    if c < 0:
                        c_idx = -c
                    elif c >= n:
                        c_idx = 2 * n - 2 - c
                    else:
                        c_idx = c

                    padded_img[i, j] = self.img[r_idx, c_idx]

        return padded_img

    def choose_kernel(self, name: str, border_mode: str = 'replicate'):
        """
        Thực hiện phép tích chập không gian (Spatial Convolution) bằng vòng lặp thủ công.
        Công thức: Pixel_mới(i,j) = Tổng_mọi_k [ Padded_Image(i+ki, j+kj) * Kernel(ki, kj) ]

        :param name: Tên bộ lọc ('Mean', 'MeanWeight', 'Gaussian')
        :param border_mode: Phương pháp xử lý biên
        :return: Ảnh đã qua xử lý lọc (dạng số nguyên uint8 [0-255])
        """
        # Gán ma trận Kernel tương ứng dựa vào tên truyền vào
        if name == 'Mean':
            kernel = self.MeanKernel33
        elif name == 'MeanWeight':
            kernel = self.MeanWeightKernel33
        elif name == 'Gaussian':
            kernel = self.GaussianKernel33
        else:
            return self.img  # Nếu nhập sai tên thì trả lại ảnh gốc

        # 1. Tạo ảnh đã được mở rộng viền
        padded_img = self._apply_padding(border_mode=border_mode)

        m, n = self.img.shape
        img_new = np.zeros([m, n], dtype=float)  # Mảng lưu kết quả ảnh mới

        # 2. Duyệt qua từng pixel (i, j) của ảnh gốc
        for i in range(m):
            for j in range(n):
                val = 0.0
                # 3. Quét qua cửa sổ 3x3 của Kernel (ki: 0..2, kj: 0..2)
                for ki in range(3):
                    for kj in range(3):
                        # Nhân giá trị pixel trên ảnh đệm với trọng số tương ứng trên Kernel rồi cộng dồn
                        val += padded_img[i + ki, j + kj] * kernel[ki, kj]

                img_new[i, j] = val

        # 4. Hạn chế giá trị nằm trong khoảng [0, 255] để tránh lỗi tràn số khi ép kiểu
        img_new = np.clip(img_new, 0, 255).astype(np.uint8)
        return img_new

    def Median33(self, border_mode: str = 'replicate'):
        """
        Bộ lọc Trung vị (Median Filter) 3x3 - Là bộ lọc phi tuyến:
        Thay thế giá trị pixel trung tâm bằng GIÁ TRỊ TRUNG VỊ của 9 pixel lân cận.
        Rất hiệu quả trong việc loại bỏ 'Nhiễu muối tiêu' (Salt-and-Pepper noise) mà vẫn giữ nét cạnh.
        """
        padded_img = self._apply_padding(border_mode=border_mode)
        m, n = self.img.shape
        img_new = np.zeros([m, n], dtype=np.uint8)

        # Duyệt qua từng vị trí pixel trên ảnh
        for i in range(m):
            for j in range(n):
                temp = []
                # Trích xuất toàn bộ 9 giá trị trong cửa sổ lân cận 3x3
                for ki in range(3):
                    for kj in range(3):
                        temp.append(padded_img[i + ki, j + kj])

                # Sắp xếp mảng 9 phần tử theo thứ tự tăng dần
                temp.sort()

                # Lấy phần tử nằm chính giữa (phần tử thứ 5, chỉ số index = 4)
                img_new[i, j] = int(temp[4])

        return img_new

    def filtering_Max_Min(self, name: str, border_mode: str = 'replicate'):
        """
        Bộ lọc Min / Max 3x3 - Bộ lọc phi tuyến dựa trên thứ tự xếp hạng:
        - Lọc Min: Chọn giá trị nhỏ nhất trong vùng 3x3 -> Làm tối ảnh, loại bỏ điểm nhiễu sáng.
        - Lọc Max: Chọn giá trị lớn nhất trong vùng 3x3 -> Làm sáng ảnh, loại bỏ điểm nhiễu tối.
        """
        padded_img = self._apply_padding(border_mode=border_mode)
        m, n = self.img.shape
        img_new = np.zeros([m, n], dtype=np.uint8)

        for i in range(m):
            for j in range(n):
                # Gom 9 điểm ảnh lân cận vào danh sách
                temp = []
                for ki in range(3):
                    for kj in range(3):
                        temp.append(padded_img[i + ki, j + kj])

                # Chọn giá trị lớn nhất hoặc nhỏ nhất theo yêu cầu
                if name == 'max':
                    img_new[i, j] = int(max(temp))  # Lấy pixel sáng nhất
                elif name == 'min':
                    img_new[i, j] = int(min(temp))  # Lấy pixel tối nhất
                else:
                    return self.img

        return img_new


# ==============================================================================
# CHƯƠNG TRÌNH CHÍNH (MAIN EXECUTION)
# ==============================================================================
if __name__ == '__main__':
    # Đọc ảnh từ đĩa ở chế độ ảnh xám (Tham số 0 tương đương cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('./images/test2.tif', 0)

    # Kiểm tra an toàn: Nếu đường dẫn tệp sai hoặc không có ảnh, tự động sinh ma trận ngẫu nhiên 100x100
    if image is None:
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # Chọn loại xử lý viền mặc định cho tất cả phép lọc: 'replicate', 'zero', hoặc 'reflect'
    BORDER_TYPE = 'replicate'

    # Khởi tạo cửa sổ Matplotlib kích thước 20x9 inch, thiết lập lưới 2 hàng, 4 cột (tổng cộng 8 ô chứa ảnh)
    fig = plt.figure(figsize=(20, 9))
    ax = fig.subplots(2, 4)

    # ------------------ HÀNG 1: CÁC BỘ LỌC TUYẾN TÍNH ------------------
    # Ô [0, 0]: Hiển thị ảnh gốc chưa qua lọc
    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title("Original")

    # Khởi tạo đối tượng lowpass_filter với convolution=True để sẵn sàng lọc tích chập
    img_filter = lowpass_filter(image, convolution=True)

    # Ô [0, 1]: Lọc Trung bình (Mean Filter)
    ax[0, 1].imshow(img_filter.choose_kernel('Mean', border_mode=BORDER_TYPE), cmap="gray")
    ax[0, 1].set_title(f"Mean filter ({BORDER_TYPE})")

    # Ô [0, 2]: Lọc Trung bình có trọng số (Weighted Mean)
    ax[0, 2].imshow(img_filter.choose_kernel('MeanWeight', border_mode=BORDER_TYPE), cmap="gray")
    ax[0, 2].set_title(f"Weighted mean ({BORDER_TYPE})")

    # Ô [0, 3]: Lọc Gaussian
    ax[0, 3].imshow(img_filter.choose_kernel('Gaussian', border_mode=BORDER_TYPE), cmap="gray")
    ax[0, 3].set_title(f"Gaussian filter ({BORDER_TYPE})")

    # ------------------ HÀNG 2: CÁC BỘ LỌC PHI TUYẾN ------------------
    # Khởi tạo đối tượng lowpass_filter cho các thuật toán lọc không dùng Kernel tuyến tính
    img_objectnew = lowpass_filter(image)

    # Ô [1, 0]: Lọc Trung vị (Median Filter)
    ax[1, 0].imshow(img_objectnew.Median33(border_mode=BORDER_TYPE), cmap="gray")
    ax[1, 0].set_title(f"Median filter ({BORDER_TYPE})")

    # Ô [1, 1]: Lọc Min (Min Filter)
    ax[1, 1].imshow(img_objectnew.filtering_Max_Min('min', border_mode=BORDER_TYPE), cmap="gray")
    ax[1, 1].set_title(f"Min filter ({BORDER_TYPE})")

    # Ô [1, 2]: Lọc Max (Max Filter)
    ax[1, 2].imshow(img_objectnew.filtering_Max_Min('max', border_mode=BORDER_TYPE), cmap="gray")
    ax[1, 2].set_title(f"Max filter ({BORDER_TYPE})")

    # Ô [1, 3]: Ô trống thứ 8 -> Ẩn trục tọa độ đi cho giao diện cân đối, đẹp mắt
    ax[1, 3].axis('off')

    # Lưu toàn bộ hình lưới chứa 7 bức ảnh kết quả ra file PDF 'filter.pdf'
    plt.savefig("filter.pdf", bbox_inches='tight')

    # Bật cửa sổ đồ họa hiển thị kết quả trực quan ra màn hình
    plt.show()