import matplotlib.pyplot as plt
import cv2
import numpy as np

class highpass_filter(object):
    """
    Lớp triển khai các bộ lọc tần số cao (High-pass Filter / Sharpening Filter):
    - Dùng để TÁCH BIÊN (Edge Detection) và LÀM SẮC NÉT ÍCH (Image Sharpening).
    """

    def __init__(self, image) -> None:
        """
        Phương thức khởi tạo (Constructor).
        :param image: Ma trận ảnh đầu vào dạng ảnh xám 2D (Gray-scale)
        """
        self.img = image

    def init_kernel(self, kernel_name: str):
        """
        Hàm định tuyến (Router): Nhận tên thuật toán lọc và gọi phương thức xử lý tương ứng.
        """
        if kernel_name == "RobertCross":
            return self.show_SharpenedRobert()
        elif kernel_name == "RobertCross1":
            return self.show_RobertCrossGradient1()
        elif kernel_name == "RobertCross2":
            return self.show_RobertCrossGradient2()
        elif kernel_name == "Sobel":
            return self.show_SharpenedSobel()
        elif kernel_name == "SobelX":
            return self.show_SobelX()
        elif kernel_name == "SobelY":
            return self.show_Sobely()
        elif kernel_name == "Laplacian":
            return self.show_SharpenedLaplacian()
        elif kernel_name == "LaplacianV1":
            return self.show_SharpenedLaplacianV1()
        elif kernel_name == "LaplacianV2":
            return self.show_SharpenedLaplacianV2()
        elif kernel_name == "LaplacianV3":
            return self.show_SharpenedLaplacianV3()
        else:
            return self.img

    def Convolution2D(self, kernel):
        """
        Thực hiện phép tích chập không gian 2D (Spatial Convolution) với Kernel 3x3 thủ công.
        """
        m, n = self.img.shape
        img_new = np.zeros([m, n])

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = self.img[i - 1, j - 1] * kernel[0, 0] \
                       + self.img[i, j - 1] * kernel[0, 1] \
                       + self.img[i + 1, j - 1] * kernel[0, 2] \
                       + self.img[i - 1, j] * kernel[1, 0] \
                       + self.img[i, j] * kernel[1, 1] \
                       + self.img[i + 1, j] * kernel[1, 2] \
                       + self.img[i - 1, j + 1] * kernel[2, 0] \
                       + self.img[i, j + 1] * kernel[2, 1] \
                       + self.img[i + 1, j + 1] * kernel[2, 2]

                img_new[i, j] = temp

        img_new = img_new.astype(np.uint8)
        return img_new

    # ==========================================================================
    # 1. BỘ LỌC ROBERT CROSS GRADIENT
    # ==========================================================================
    def show_RobertCrossGradient1(self):
        G_cross1 = np.array(([0, 0, 0], [0, -1, 0], [0, 0, 1]), dtype="float")
        return self.Convolution2D(G_cross1)

    def show_RobertCrossGradient2(self):
        G_cross2 = np.array(([0, 0, 0], [0, 0, -1], [0, 1, 0]), dtype="float")
        return self.Convolution2D(G_cross2)

    def show_SharpenedRobert(self):
        return self.show_RobertCrossGradient1() + self.show_RobertCrossGradient2() + self.img

    # ==========================================================================
    # 2. BỘ LỌC SOBEL
    # ==========================================================================
    def show_SobelX(self):
        SobelX = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype="float")
        return self.Convolution2D(SobelX)

    def show_Sobely(self):
        Sobely = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="float")
        return self.Convolution2D(Sobely)

    def show_SharpenedSobel(self):
        return self.show_SobelX() + self.show_Sobely() + self.img

    # ==========================================================================
    # 3. BỘ LỌC LAPLACIAN
    # ==========================================================================
    def show_SharpenedLaplacian(self):
        Laplacian_kerner = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="float")
        return self.img - self.Convolution2D(Laplacian_kerner)

    def show_SharpenedLaplacianV1(self):
        Laplacian_kerner = np.array(([1, 1, 1], [1, -8, 1], [1, 1, 1]), dtype="float")
        return self.img - self.Convolution2D(Laplacian_kerner)

    def show_SharpenedLaplacianV2(self):
        Laplacian_kerner = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]), dtype="float")
        return self.img + self.Convolution2D(Laplacian_kerner)

    def show_SharpenedLaplacianV3(self):
        Laplacian_kerner = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype="float")
        return self.img + self.Convolution2D(Laplacian_kerner)


if __name__ == '__main__':
    # Đọc ảnh gốc dạng ảnh xám từ đường dẫn
    image = cv2.imread('./images/len.tif', 0)
    image2 = cv2.imread('./images/laplapcian.tif', 0)

    # Kiểm tra xem tệp ảnh có mở thành công không
    if image is None:
        print("Lỗi: Không tìm thấy ảnh tại đường dẫn './images/len.tif'. Hãy kiểm tra lại thư mục lưu ảnh.")
    else:
        hp = highpass_filter(image)
        hp2 = highpass_filter(image2)

        # ----------------------------------------------------------------------
        # CỬA SỔ 1: KẾT QUẢ BỘ LỌC ROBERT CROSS (XUẤT FILE filter_Robert.pdf)
        # ----------------------------------------------------------------------
        fig1, ax1 = plt.subplots(2, 2, figsize=(9, 9))

        # Ô [0, 0]: Ảnh gốc
        ax1[0, 0].set_title("Original")
        ax1[0, 0].imshow(image, cmap="gray")

        # Ô [0, 1]: Robert hướng 1
        Robertcross1 = hp.init_kernel("RobertCross1")
        ax1[0, 1].imshow(Robertcross1, cmap='gray')
        ax1[0, 1].set_title("Anh loc theo huong thu 1")

        # Ô [1, 1]: Robert hướng 2
        Robertcross2 = hp.init_kernel("RobertCross2")
        ax1[1, 1].imshow(Robertcross2, cmap='gray')
        ax1[1, 1].set_title("Anh loc theo huong thu 2")

        # Ô [1, 0]: Cải thiện bằng Robert
        Robertcross = hp.init_kernel("RobertCross")
        ax1[1, 0].imshow(Robertcross, cmap='gray')
        ax1[1, 0].set_title("Anh cai thien bang bo loc Robert")

        # Lưu cửa sổ 1 ra PDF
        fig1.savefig("filter_Robert.pdf", bbox_inches='tight')

        # CỬA SỔ 2: KẾT QUẢ BỘ LỌC SOBEL (XUẤT FILE filter_Sobel.pdf)

        fig2, ax2 = plt.subplots(2, 2, figsize=(9, 9))

        # Ô [0, 0]: Ảnh gốc
        ax2[0, 0].set_title("Original")
        ax2[0, 0].imshow(image, cmap="gray")

        # Ô [0, 1]: Sobel theo hướng X
        sobel_x = hp.init_kernel("SobelX")
        ax2[0, 1].imshow(sobel_x, cmap='gray')
        ax2[0, 1].set_title("Anh loc Sobel theo huong X")

        # Ô [1, 1]: Sobel theo hướng Y
        sobel_y = hp.init_kernel("SobelY")
        ax2[1, 1].imshow(sobel_y, cmap='gray')
        ax2[1, 1].set_title("Anh loc Sobel theo huong Y")

        # Ô [1, 0]: Cải thiện bằng Sobel
        sobel_sharpened = hp.init_kernel("Sobel")
        ax2[1, 0].imshow(sobel_sharpened, cmap='gray')
        ax2[1, 0].set_title("Anh cai thien bang bo loc Sobel")

        # Lưu cửa sổ 2 ra PDF
        fig2.savefig("filter_Sobel.pdf", bbox_inches='tight')

        # ----------------------------------------------------------------------
        # CỬA SỔ 3: BỘ LỌC LAPLACIAN (XUẤT FILE filter_Laplacian.pdf)
        # ----------------------------------------------------------------------
        # Tạo lưới 2x3 để hiển thị đủ ảnh gốc + 4 biến thể của Laplacian
        fig3, ax3 = plt.subplots(2, 3, figsize=(15, 9))

        # Ô [0, 0]: Ảnh gốc
        ax3[0, 0].set_title("Original")
        ax3[0, 0].imshow(image2, cmap="gray")

        # Ô [0, 1]: Laplacian chuẩn (Tâm -4)
        laplacian_std = hp2.init_kernel("Laplacian")
        ax3[0, 1].imshow(laplacian_std, cmap='gray')
        ax3[0, 1].set_title("Laplacian Chuan (Tam -4)")

        # Ô [0, 2]: Laplacian V1 (Tâm -8)
        laplacian_v1 = hp2.init_kernel("LaplacianV1")
        ax3[0, 2].imshow(laplacian_v1, cmap='gray')
        ax3[0, 2].set_title("Anh loc Laplacian Bien The V1 (Tam -8)")

        ax3[1, 0].axis('off')

        # Ô [1, 1]: Laplacian V2 (Tâm +4)
        laplacian_v2 = hp2.init_kernel("LaplacianV2")
        ax3[1, 1].imshow(laplacian_v2, cmap='gray')
        ax3[1, 1].set_title("Anh loc Laplacian Bien The 2 (Tam +4)")

        # Ô [1, 2]: Laplacian V3 (Tâm +8)
        laplacian_v3 = hp2.init_kernel("LaplacianV3")
        ax3[1, 2].imshow(laplacian_v3, cmap='gray')
        ax3[1, 2].set_title("Anh loc Laplacian Bien The 3 (Tam +8)")

        # Lưu cửa sổ Laplacian ra PDF
        fig3.savefig("filter_Laplacian.pdf", bbox_inches='tight')

        # Hiển thị cả 3 cửa sổ cùng lúc ra màn hình
        plt.show()