To run Q2 for seam carving: 
python seam_carving.py 
* This runs the main block, which executes for all 3 images and their desired sizer
* For each image and desired size, the image will be resized using seam carving, cropping, and scaling
* Saves all results in the images/ directory
* NOTE: this uses numba to speed up the computation, without numba this will take a while... 


To run Q3 for corner detection:
python corner_detection.py
* Main block will compute eigenvalues and detect corners for both images in assignment
* The code will do this for std=2, 10 and save all results in the images/ directory
