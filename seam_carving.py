import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def load_image(path):
    img = cv2.imread(path)[..., ::-1]
    return img 


def get_gradient(img):
    img_grey = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)

    Ix = cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_grey, cv2.CV_64F, 0, 1, ksize=3)

    img_grad = np.sqrt(Ix**2 + Iy**2)

    return img_grad 


def display_image(img, cmap=None):
    plt.cla()
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)

    

def seam_carve(img_grad):
    pass 