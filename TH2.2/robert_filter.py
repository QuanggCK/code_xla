import matplotlib.pyplot as plt
import cv2
import numpy as np
plt.rcParams.update({"text.usetex":True})
class highpass_filter(object):
    '''Các bộ lọc cơ bản trong highpass filter'''
    def __init__(self,image) -> None:
        '''constructor: 
            - Ảnh đầu vào
        '''
        self.img=image
    def init_kernel(self, kernel_name:str):
        '''
        Nhập vào tên kernel:
            - RobertCross: xem kết quả lọc sắc nét bằng bộ lọc RobertCrossGradient
            - RobertCross1: xem Lọc RobertCross theo hướng 1
            - RobertCross2: xem Lọc RobertCross theo hướng 2
        '''
        if kernel_name=="RobertCross":
            return self.show_SharpenedRobert()
        elif kernel_name=="RobertCross1":
            return self.show_RobertCrossGradient1()
        elif kernel_name=="RobertCross2":
            return self.show_RobertCrossGradient2()
    def Convolution2D(self,kernel):
        m, n = self.img.shape
        img_new = np.zeros([m, n])
        for i in range(1, m-1):
            for j in range(1, n-1):
                temp=  self.img[i-1, j-1]    * kernel[0, 0]\
                    +  self.img[i, j-1]      * kernel[0, 1]\
                    +  self.img[i+1, j - 1]  * kernel[0, 2]\
                    +  self.img[i-1, j]      * kernel[1, 0]\
                    +  self.img[i, j]        * kernel[1, 1]\
                    +  self.img[i+1, j]      * kernel[1, 2]\
                    +  self.img[i - 1, j+1]  * kernel[2, 0]\
                    +  self.img[i, j + 1]    * kernel[2, 1]\
                    +  self.img[i + 1, j + 1]* kernel[2, 2]
                img_new[i, j]= temp
        img_new = img_new.astype(np.uint8)
        return img_new
    def show_RobertCrossGradient1(self):
        '''Kết quả RobertCrossGradient theo hướng thứ nhất'''
        G_cross1 = np.array(([0, 0, 0], [0,-1, 0], [0, 0, 1]), dtype="float")
        return self.Convolution2D(G_cross1)
    def show_RobertCrossGradient2(self):
        '''Kết quả RobertCrossGradient theo hướng thứ nhất'''
        G_cross2 = np.array(([0, 0, 0], [0, 0,-1], [0, 1, 0]), dtype="float")
        return self.Convolution2D(G_cross2)
    def show_SharpenedRobert(self):
        '''Kết quả Lọc sắc nét sử dụng bộ lọc RobertCrossGradient theo 2 hướng chéo'''
        return self.show_RobertCrossGradient1()+self.show_RobertCrossGradient2()+self.img

    
if __name__=='__main__':
    # Đọc và hiển thị ảnh gốc
    
    image = cv2.imread('images/len.tif', 0)

    fig=plt.figure(figsize=(9, 9))
    ax=fig.subplots(2,2)
    # hien thi anh goc 
    ax[0,0].set_title("Orignal")
    ax[0,0].imshow(image,cmap="gray")
    ## Su dung bo loc RobertCrossGradient
    # hien thi anh loc Robert theo huong thu 1 
    Robertcross1=highpass_filter(image).init_kernel("RobertCross1")
    ax[0,1].imshow(Robertcross1,cmap='gray')
    ax[0,1].set_title("Anh loc theo huong thu 1")
    # hien thi anh loc Robert theo huong thu 2 
    Robertcross2=highpass_filter(image).init_kernel("RobertCross2")
    ax[1,1].imshow(Robertcross2,cmap='gray')
    ax[1,1].set_title("Anh loc theo huong thu 2")
    # Ket qua anh loc sac net bang bo loc Robert
    Robertcross=highpass_filter(image).init_kernel("RobertCross")
    ax[1,0].imshow(Robertcross,cmap='gray')
    ax[1,0].set_title("Anh cai thien bang bo loc Robert")
    # plt.savefig("filter_Robert.pdf",bbox_inches='tight')
    plt.show()