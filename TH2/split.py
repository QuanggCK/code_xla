import numpy as np
import matplotlib.pyplot as plt
import cv2

def image_spliting(image):
    return cv2.split(image)

if __name__ == '__main__':
    img = cv2.imread('images/wave.png')
    fig = plt.figure()
    ax = fig.subplots(2, 2)

    ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("original")

    B, G, R = image_spliting(img)

    ax[0, 1].imshow(cv2.cvtColor(R, cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title("The Red channel")

    ax[1, 0].imshow(cv2.cvtColor(G, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title("The Green channel")

    ax[1, 1].imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title("The Blue channel")

    plt.savefig("splitting_image.pdf", bbox_inches='tight')
    plt.show()