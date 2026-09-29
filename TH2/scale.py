import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def image_scaling(image, scaleX=1,scaleY=1):
    (h,w)=image.shape[:2]
    M=np.float32([[scaleX,0, 0],
                  [0, scaleY, 0]])
    return cv2.warpAffine(image, M,(w,h))

if __name__=='__main__':
    img = plt.imread('images/trex.png')
    fig = plt.figure()
    ax1,ax2 = fig.subplots(1,2)
    ax1.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ax1.set_title("Original")

    scaled = image_scaling(img,0.2,0.2)
    ax2.imshow(cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB))
    ax2.set_title("Scaled")
    h,w=img.shape[:2]
    ax2.set_xlim(0, w * 0.2)
    ax2.set_ylim(h * 0.2, 0)

    plt.savefig("scale.pdf", bbox_inches='tight')
    plt.show()
