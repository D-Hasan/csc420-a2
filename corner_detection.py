import numpy as np 
from numba import jit 
import cv2 
import matplotlib.pyplot as plt 

def load_image(path):
    img = cv2.imread(path)[..., ::-1]
    return img 


def get_blurred_image_gradients(img):
    img_grey = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
    img_blur = cv2.GaussianBlur(img_grey,(5,5),7)
    Ix = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)

    return Ix, Iy 

def compute_M(img, k=7, std=10):

    Ix, Iy = get_blurred_image_gradients(img)

    Ix_2 = cv2.GaussianBlur(Ix * Ix,(k,k),std)
    Iy_2 = cv2.GaussianBlur(Iy * Iy,(k,k),std)
    IxIy = cv2.GaussianBlur(Ix * Iy,(k,k),std)

    return np.stack([Ix_2, IxIy, IxIy, Iy_2], axis=2)


def compute_eigenvalues(img, title):
    M = compute_M(img)

    eigenvals_list = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            m_matrix = M[i,j].reshape(2,2)
            eigenvals = np.linalg.eigvalsh(m_matrix)
            eigenvals_list.append(eigenvals)
    
    # import pdb; pdb.set_trace()
    eigenvals_array = np.stack(eigenvals_list)
    plt.scatter(eigenvals_array[:,0], eigenvals_array[:,1])
    plt.xlim(eigenvals_array.min(), eigenvals_array.max())
    plt.ylim(eigenvals_array.min(), eigenvals_array.max())
    plt.title(title)
    plt.show()


def display_corners(img, threshold=0.33*1e7, k=7, std=10):
    M = compute_M(img, k=k, std=std)
    eigensum = np.zeros((img.shape[0], img.shape[1], 2))
    corner_x = []
    corner_y = []

    eigenvals_list = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            m_matrix = M[i,j].reshape(2,2)
            eigenvals = np.linalg.eigvalsh(m_matrix)
            eigensum[i,j] = eigenvals
            if (eigenvals > threshold).all():
                corner_x.append(j)
                corner_y.append(i)

    img2 = img.copy()
    img2[(eigensum[:,:,0] > threshold) & (eigensum[:,:,1] > threshold)] = [255, 255, 0]
    print(((eigensum[:,:,0] > threshold) & (eigensum[:,:,1] > threshold)).sum())
    plt.imshow(img2)
    plt.scatter(corner_x, corner_y, s=12, c='r')





    

