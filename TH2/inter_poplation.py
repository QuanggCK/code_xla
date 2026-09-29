# Code minh hoa cho resize anh theo chieu width hoac height

import numpy as np
import matplotlib.pyplot as plt
import cv2


def image_resized(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if height is None:
        # we only resized the image by specifying the width
        r = width / float(w)

        # From the image width, we compute the ratio of the new height to the
        # old height, we simply define our ratio r to be the new width divided
        # by the old width
        dim = (width, int(h * r))

    else:
        # we only resized the image by specifying the height
        r = height / float(h)
        dim = (int(w * r), height)

    resize = cv2.resize(image, dim, interpolation=inter)
    return resize


if __name__ == '__main__':
    img = cv2.imread('./images/trex.png')

    fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2)

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original")

    resized = image_resized(img, width=50)

    ax2.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    ax2.set_title("Resize image with INTER_AREA method")

    plt.savefig("Resize_image.pdf", bbox_inches='tight')
    plt.show()