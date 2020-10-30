import numpy as np 
from numba import jit 
import cv2 
import matplotlib.pyplot as plt 

def load_image(path):
    img = cv2.imread(path)[..., ::-1]
    return img 


def get_blurred_image_gradients(img):
    img_grey = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
    img_grey = img_grey*1.1
    img_blur = cv2.GaussianBlur(img_grey,(5,5),7)
    Ix = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)

    return Ix, Iy 

def compute_M(img, k=7, std=10):

    Ix, Iy = get_blurred_image_gradients(img)

    Ix_2 = cv2.GaussianBlur(np.multiply(Ix,Ix),(k,k),std)
    Iy_2 = cv2.GaussianBlur(np.multiply(Iy,Iy),(k,k),std)
    IxIy = cv2.GaussianBlur(np.multiply(Ix,Iy),(k,k),std)

    return np.stack([Ix_2, IxIy, IxIy, Iy_2], axis=2)


def compute_eigenvalues(img_path, title, k=7, std=10):
    ''' Detects corners using Harris corner detector algo.
        
        img_path (str): path to image
        title (str): Prefix for plot title
        k (int): kernel size
        std (int): std for Gaussian kernel, used for window function
    '''
    img = load_image(img_path)
    M = compute_M(img, k=k, std=std)

    eigenvals_list = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            m_matrix = M[i,j].reshape(2,2)
            eigenvals = np.linalg.eigvalsh(m_matrix)
            eigenvals_list.append(eigenvals)
    
    # import pdb; pdb.set_trace()
    eigenvals_array = np.stack(eigenvals_list)
    
    plt.clf()
    plt.cla()
    plt.scatter(eigenvals_array[:,0], eigenvals_array[:,1])
    plt.xlim(eigenvals_array.min(), eigenvals_array.max())
    plt.ylim(eigenvals_array.min(), eigenvals_array.max())
    plt.title('{}, $\sigma=${}'.format(title, std))
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$')

    save_path = 'images/std{}_eigen_'.format(std) + img_path
    plt.savefig(save_path)


def detect_corners(img_path, title,threshold=0.5, k=7, std=10, alpha=0.05):
    ''' Detects corners using Harris corner detector algo.
        
        img_path (str): path to image
        title (str): Prefix for plot title
        threshold (float): [0, 1] and gets multiplied by 10^7
        k (int): kernel size
        std (int): std for Gaussian kernel, used for window function
        alpha (float): for R
    '''
    img = load_image(img_path)
    M = compute_M(img, k=k, std=std)
    eigensum = np.zeros((img.shape[0], img.shape[1], 2))

    R_lambda = lambda x1, x2: x1*x2 - alpha*(x1+x2)**2
    R_threshold = R_lambda(threshold*1e7, threshold*1e7)
    corner_x = []
    corner_y = []

    for i in range(len(img)):
        for j in range(len(img[0])):
            m_matrix = M[i,j].reshape(2,2)
            eigenvals = np.linalg.eigvalsh(m_matrix)
            eigensum[i,j] = eigenvals

            R = R_lambda(eigenvals[0], eigenvals[1])
            if R > R_threshold:
                corner_x.append(i)
                corner_y.append(j)

    img2 = img.copy()
    plt.clf()
    plt.cla()
    plt.imshow(img2)
    plt.scatter(corner_y, corner_x, s=12, c='r')
    plt.title('{}, $\sigma=${}, threshold={}$*10^7$'.format(title, std, threshold))
    save_path = 'images/std{}_corners_'.format(std) + img_path
    plt.savefig(save_path)

if __name__ == '__main__':
    img_path1 = 'u_coll_1.jpg'
    img_path2 = 'u_coll_2.jpg'

    # Corner detection with sigma=10
    compute_eigenvalues(img_path1, 'Image 1', std=10)
    compute_eigenvalues(img_path2, 'Image 1', std=10)

    detect_corners(img_path1, 'Image 1', threshold=0.5, std=10)    
    detect_corners(img_path2, 'Image 2', threshold=0.5, std=10)    

    # Corner detection with sigma=2
    compute_eigenvalues(img_path1, 'Image 1', std=2)
    compute_eigenvalues(img_path2, 'Image 2', std=2)

    detect_corners(img_path1, 'Image 1', threshold=0.45, std=2)    
    detect_corners(img_path2, 'Image 2', threshold=0.45, std=2) 

