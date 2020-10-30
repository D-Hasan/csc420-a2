import numpy as np 
from numba import jit
import matplotlib.pyplot as plt 
import cv2 

def load_image(path):
    ''' Load image and swap to RGB'''
    img = cv2.imread(path)[..., ::-1]
    return img 


def get_gradient(img):
    ''' Calculate magnitude of image gradient in greyscale'''
    img_grey = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
    img_blur = cv2.GaussianBlur(img_grey,(5,5),7)
    Ix = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)

    img_grad = np.sqrt(Ix**2 + Iy**2)

    return img_grad 


def display_image(img, cmap=None):
    ''' Display image with matplotlib'''
    plt.cla()
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)


@jit(nopython=True)
def remove_seam(img_grad, img):
    ''' Remove seam with minimum cumulative energy'''
    h, w = img_grad.shape
    energy = np.zeros(img_grad.shape)

    # base case, energy at first row is the same as grad. magn.
    energy[0, :] = img_grad[0,:]

    for i in range(1, h):
        for j in range(w):
            parent_min, parent_max = max(j-1, 0), min(j+1, w) 
            energy[i,j] = img_grad[i,j] + energy[i-1, parent_min:parent_max+1].min()

    min_col = np.argmin(energy[-1,:])
    min_path = [min_col]
    for i in range(h-1, 0, -1):
        parent_min, parent_max = max(min_col-1, 0), min(min_col+1, w) 
        min_col = max(min_col-1, 0) + energy[i, parent_min:parent_max+1].argmin()
        min_path.append(min_col)

    carved_img = np.zeros((h, w-1, 3))
    for row, column in enumerate(min_path):
        carved_img[row, :column] = img[row, :column]
        carved_img[row, column:] = img[row, column+1:]

    return carved_img


def seam_carve(img, desired_size):
    ''' Scale image with content awareness using seam carving algorithm'''
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


def crop(img, desired_size):
    ''' Crop image equally from all sides to desired size'''
    img2 = img.copy()

    h, w, _ = img.shape 
    new_h, new_w = desired_size

    crop_margin_h = (h-new_h)//2
    crop_margin_w = (w-new_w)//2 

    cropped = img2 

    if crop_margin_h > 0:
        cropped = img2[crop_margin_h:-crop_margin_h]
    if crop_margin_w > 0:
        cropped = cropped[:,crop_margin_w:-crop_margin_w]

    return cropped 

def scale(img, desired_size):
    ''' Scale image to desired size using linear interpolation'''
    new_h, new_w = desired_size
    return cv2.resize(img, (new_w, new_h))


def run_end_to_end(img_path, img_num, desired_size):
    ''' Transform image to desired size with seam carving, cropping, and scaling'''
    carved_path = 'images/carved_{}_{}'.format(img_num, img_path)
    crop_path = 'images/crop_{}_{}'.format(img_num, img_path)
    scale_path = 'images/scale_{}_{}'.format(img_num, img_path)
    
    img = load_image(img_path)
    
    carved_img = seam_carve(img, desired_size)
    cv2.imwrite(carved_path, cv2.cvtColor(carved_img.astype('float32'), cv2.COLOR_RGB2BGR))
    
    crop_img = crop(img, desired_size)
    cv2.imwrite(crop_path, cv2.cvtColor(crop_img.astype('float32'), cv2.COLOR_RGB2BGR))

    scale_img = scale(img, desired_size)
    cv2.imwrite(scale_path, cv2.cvtColor(scale_img.astype('float32'), cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # Image 1
    img_path = 'ex1.jpg'
    img_num = 'a'
    desired_size = (968, 957)
    run_end_to_end(img_path, img_num, desired_size)
    print('Finished Image 1a -> {}'.format(desired_size))


    # Image 2 a
    img_path = 'ex2.jpg'
    img_num = 'a'
    desired_size = (961, 1200)
    run_end_to_end(img_path, img_num, desired_size)
    print('Finished Image 2a -> {}'.format(desired_size))


    # # Image 2 b
    img_path = 'ex2.jpg'
    img_num = 'b'
    desired_size = (861, 1200)
    run_end_to_end(img_path, img_num, desired_size)
    print('Finished Image 2b -> {}'.format(desired_size))


    # # Image 3 a
    img_path = 'ex3.jpg'
    img_num = 'a'
    desired_size = (870, 1440)
    run_end_to_end(img_path, img_num, desired_size)
    print('Finished Image 3a -> {}'.format(desired_size))

    # # Image 3 b
    img_path = 'ex3.jpg'
    img_num = 'b'
    desired_size = (870, 1200)
    run_end_to_end(img_path, img_num, desired_size)
    print('Finished Image 3b -> {}'.format(desired_size))
