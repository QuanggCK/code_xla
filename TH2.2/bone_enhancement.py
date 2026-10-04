import numpy as np
import matplotlib.pyplot as plt
import cv2

def Convolution2D(img, kernel):
    """
    Thực hiện phép convolution giữa ảnh và kernel 3x3.
    Không mở rộng pixel ở biên.
    """
    m, n = img.shape
    # Tạo ảnh mới cùng kích thước ảnh gốc
    img_new = np.zeros([m, n], dtype=np.float32)

    # Duyệt các pixel bên trong ảnh
    # Bỏ qua pixel biên
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            # Lấy vùng 3x3 xung quanh pixel [i,j]
            region = img[i - 1:i + 2, j - 1:j + 2]
            # Nhân vùng ảnh với kernel rồi cộng lại
            img_new[i, j] = np.sum(region * kernel)
    return img_new



def laplacian_filter(img):
    """
    Bộ lọc Laplacian chuẩn.
    Dùng để phát hiện các vùng có sự thay đổi cường độ mạnh,
    đặc biệt là các cạnh trong ảnh.
    """

    Laplacian_kernel = np.array([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0]
    ], dtype=np.float32)

    return Convolution2D(img, Laplacian_kernel)

def sharpen_laplacian(image, laplacian):
    """
    Kết hợp ảnh gốc với ảnh Laplacian
    để làm sắc nét ảnh.
    """
    sharpened = image - laplacian
    # Đưa giá trị về khoảng 0 - 255
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened

def sobel_gradient(img):
    """
    Tính Sobel theo hướng X và Y,
    sau đó tính độ lớn gradient.
    """

    # Kernel Sobel theo hướng X
    sobel_x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Kernel Sobel theo hướng Y
    sobel_y_kernel = np.array([
        [-1, -2, -1],
        [0,   0,  0],
        [1,   2,  1]
    ], dtype=np.float32)

    # Tính gradient theo X
    gx = Convolution2D(img, sobel_x_kernel)

    # Tính gradient theo Y
    gy = Convolution2D(img, sobel_y_kernel)

    # Tính độ lớn gradient
    gradient = np.sqrt(gx ** 2 + gy ** 2)

    # Chuẩn hóa về 0 - 255
    gradient = cv2.normalize(
        gradient,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gradient


def smooth_sobel(sobel_image):
    """
    Làm trơn ảnh Sobel bằng bộ lọc trung bình 5x5.
    """

    # Kernel 5x5
    box_kernel = np.ones((5, 5), dtype=np.float32)

    # Chuẩn hóa kernel
    box_kernel = box_kernel / (5 * 5)

    # Dùng cv2.filter2D để xử lý kernel lớn 5x5
    smooth = cv2.filter2D(
        sobel_image,
        -1,
        box_kernel
    )

    return smooth


def create_mask(laplacian, smooth_sobel):
    """
    Mask được tạo bằng phép nhân:
    
        Mask = Laplacian * Smoothed Sobel

    Trước khi nhân, cả hai ảnh được chuẩn hóa về 0 - 1.
    """

    # Chuẩn hóa Laplacian
    laplacian_norm = cv2.normalize(
        np.abs(laplacian),
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    # Chuẩn hóa Sobel đã làm trơn
    sobel_norm = cv2.normalize(
        smooth_sobel,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    # Tạo mask
    mask = laplacian_norm * sobel_norm

    # Đưa mask về 0 - 255
    mask = mask * 255

    return mask


def sharpen_with_mask(image, mask):
    """
    Cộng ảnh gốc với mask:

        Sharpened = Original + Mask
    """

    sharpened = image + mask

    # Giới hạn giá trị trong khoảng 0 - 255
    sharpened = np.clip(sharpened, 0, 255)

    return sharpened


def power_law(image, gamma=0.5):
    """
    Thực hiện biến đổi Power-law:

        s = c * r^gamma

    Ở đây chọn c = 1.
    """

    # Chuẩn hóa ảnh về 0 - 1
    normalized = image / 255.0

    # Power-law transformation
    result = np.power(normalized, gamma)

    # Đưa lại về 0 - 255
    result = result * 255

    return result


if __name__ == '__main__':
    # Đọc ảnh grayscale
    image = cv2.imread("images/spooky_scary_skeleton.ppm",0)

    laplacian_image = laplacian_filter(image)

    sharpened_laplacian = sharpen_laplacian(
        image,
        laplacian_image
    )

    sobel_image = sobel_gradient(image)

    smooth_sobel_image = smooth_sobel(sobel_image)

    mask_image = create_mask(
        laplacian_image,
        smooth_sobel_image
    )

    sharpened_mask = sharpen_with_mask(
        image,
        mask_image
    )

    final_image = power_law(
        sharpened_mask,
        gamma=0.5
    )


    fig = plt.figure(figsize=(16, 9))

    ax = fig.subplots(2, 4)

    ax[0, 0].imshow(
        image,
        cmap="gray"
    )
    ax[0, 0].set_title("(a) Original")

    ax[0, 1].imshow(
        np.abs(laplacian_image),
        cmap="gray"
    )
    ax[0, 1].set_title("(b) Laplacian")

    ax[0, 2].imshow(
        sharpened_laplacian,
        cmap="gray"
    )
    ax[0, 2].set_title("(c) Sharpened image")

    ax[0, 3].imshow(
        sobel_image,
        cmap="gray"
    )
    ax[0, 3].set_title("(d) Sobel Gradient")

    ax[1, 0].imshow(
        smooth_sobel_image,
        cmap="gray"
    )
    ax[1, 0].set_title("(e) Sobel + 5x5 Box")

    ax[1, 1].imshow(
        mask_image,
        cmap="gray"
    )
    ax[1, 1].set_title("(f) Mask")

    ax[1, 2].imshow(
        sharpened_mask,
        cmap="gray"
    )
    ax[1, 2].set_title("(g) Sharpened + Mask")

    ax[1, 3].imshow(
        final_image,
        cmap="gray"
    )
    ax[1, 3].set_title("(h) Power-law")

    for i in range(2):
        for j in range(4):
            ax[i, j].axis("off")


    plt.tight_layout()

    plt.savefig("bone_enhanced.png",bbox_inches="tight")
    plt.show()