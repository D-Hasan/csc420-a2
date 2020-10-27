import numpy as np 
from numba import jit
import matplotlib.pyplot as plt 
import cv2 

def load_image(path):
    img = cv2.imread(path)[..., ::-1]
    # import pdb; pdb.set_trace()
    return img 


def get_gradient(img):
    img_grey = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
    img_blur = cv2.GaussianBlur(img_grey,(5,5),7)
    Ix = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)

    img_grad = np.sqrt(Ix**2 + Iy**2)

    return img_grad 


def display_image(img, cmap=None):
    plt.cla()
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)

@jit(nopython=True)
def find_min_path(energy):
    h, w = energy.shape 
    min_col = np.argmin(energy[-1,:])
    min_path = [min_col]
    for i in range(h-1, 0, -1):
        parent_min, parent_max = max(min_col-1, 0), min(min_col+1, w) 
        min_col = max(min_col-1, 0) + energy[i, parent_min:parent_max+1].argmin()
        min_path.append(min_col)

    return min_path

@jit(nopython=True)
def remove_seam(img_grad, img):
    h, w = img_grad.shape
    energy = np.zeros(img_grad.shape)

    # base case, energy at first row is the same as grad. magn.
    energy[0, :] = img_grad[0,:]

    for i in range(1, h):
        for j in range(w):
            parent_min, parent_max = max(j-1, 0), min(j+1, w) 
            energy[i,j] = img_grad[i,j] + energy[i-1, parent_min:parent_max+1].min()

    min_path = find_min_path(energy)
    carved_img = np.zeros((h, w-1, 3))
    for row, column in enumerate(min_path):
        carved_img[row, :column] = img[row, :column]
        carved_img[row, column:] = img[row, column+1:]

    return carved_img


def seam_carve(img, desired_size):
    height, width, _ = img.shape 
    desired_height, desired_width = desired_size

    img_grad = get_gradient(img)

    carved_img = img 
    for i in range(width - desired_width):
        carved_img_grad = get_gradient(carved_img)
        carved_img = remove_seam(carved_img_grad, carved_img)
        # print('{} of {} carved'.format(i+1, width-desired_width))

    carved_img = np.transpose(carved_img, (1,0,2))
    for i in range(height - desired_height):
        carved_img_grad = get_gradient(carved_img)
        carved_img = remove_seam(carved_img_grad, carved_img)
        # print('{} of {} carved'.format(i+1, height - desired_height))


    carved_img = np.transpose(carved_img, (1,0,2))

    return carved_img

if __name__ == '__main__':
    # Image 1
    # img_path = 'ex1.jpg'
    # img = load_image(img_path)
    # desired_size = (968, 957)
    # carved_img = seam_carve(img, desired_size)
    # cv2.imwrite('carved' + img_path, cv2.cvtColor(carved_img.astype('float32'), cv2.COLOR_RGB2BGR))

    # Image 2 a
    # img_path = 'ex2.jpg'
    # img = load_image(img_path)
    # desired_size = (961, 1200)
    # carved_img = seam_carve(img, desired_size)
    # cv2.imwrite('carved_a' + img_path, cv2.cvtColor(carved_img.astype('float32'), cv2.COLOR_RGB2BGR))

    # # Image 2 b
    # img_path = 'ex2.jpg'
    # img = load_image(img_path)
    # desired_size = (861, 1200)
    # carved_img = seam_carve(img, desired_size)
    # cv2.imwrite('carved_b' + img_path, cv2.cvtColor(carved_img.astype('float32'), cv2.COLOR_RGB2BGR))


    # # Image 3 a
    # img_path = 'ex3.jpg'
    # img = load_image(img_path)
    # desired_size = (870, 1440)
    # carved_img = seam_carve(img, desired_size)
    # cv2.imwrite('carved_a' + img_path, cv2.cvtColor(carved_img.astype('float32'), cv2.COLOR_RGB2BGR))


    # # Image 3 b
    img_path = 'ex3.jpg'
    img = load_image(img_path)
    desired_size = (870, 1200)
    carved_img = seam_carve(img, desired_size)
    cv2.imwrite('carved_b' + img_path, cv2.cvtColor(carved_img.astype('float32'), cv2.COLOR_RGB2BGR))