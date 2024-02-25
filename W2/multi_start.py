import cv2
import numpy as np
import multi_sol_Laplace_Equation_Axb
import os
import sys
import math
from dataclasses import dataclass

@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    iterMax: float
    tol: float


#Example script: You should replace the beginning of each function ('sol')
#with the name of your group. i.e. if your gropu name is 'G8' you should
#call :
#G8_DualTV_Inpainting_GD(I, mask, paramInp, paramROF)


# ======>>>>  input data  <<<<=======
# folder with the images
folderInput='./'


#There are several black and white images to test:
#  image1_toRestore.jpg
#  image2_toRestore.jpg
#  image3_toRestore.jpg
#  image4_toRestore.jpg
#  image5_toRestore.jpg

# figure name to process. Options: image1 to image5
figure_name = 'Image'
#figure_name = 'Image15'
# ======>>>>  process  <<<<=======

# read an image
figure_name = os.path.join(folderInput, figure_name + '.jpg')
I = cv2.imread(figure_name, cv2.IMREAD_UNCHANGED)
Iinp = np.zeros(I.shape, dtype=float)

#Number of pixels for each dimension, and number of channles
# get dimensions of image
dimensions = I.shape

# height, width, number of channels in image
height = I.shape[0]
width = I.shape[1]

print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)

# show image
cv2.imshow('image',I)
cv2.waitKey(0)

#Load the mask
mask_name = "Masked"
#mask_name = "masked15"
mask_img_name = os.path.join(folderInput, mask_name + '.jpg')
mask_img = cv2.imread(mask_img_name, cv2.IMREAD_UNCHANGED)

cv2.imshow('mask_img', mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Load the binary mask
binary_mask_name = "binary"+mask_name
binary_mask_img_name = os.path.join(folderInput, binary_mask_name + '.jpg')
binary_mask_img = cv2.imread(binary_mask_img_name, cv2.IMREAD_UNCHANGED)
if binary_mask_img.ndim == 3:
    binary_mask_img = binary_mask_img[:, :, 0]


cv2.imshow('bin_mask_img', binary_mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# height, width, number of channels in image
height_mask = mask_img.shape[0]
width_mask = mask_img.shape[1]
dimensions_mask = mask_img.shape

ni=height_mask
nj=width_mask

print('Mask Dimension    : ', dimensions_mask)
print('Mask Height       : ', height_mask)
print('Mask Width        : ', width_mask)

#We want to inpaint those areas in which mask == 1
#mask1 = mask_img >128
#mask=mask1.astype('float')

#mask(i,j) == 1 means we have lost information in that pixel
#mask(i,j) == 0 means we have information in that pixel

# Modification: We have to normalize both images (image and mask) together
global_min = min(np.min(I), np.min(mask_img),np.min(binary_mask_img))
global_max = max(np.max(I), np.max(mask_img),np.max(binary_mask_img))

I_normalized = (I - global_min) / (global_max - global_min)
mask_img_normalized = (mask_img - global_min) / (global_max - global_min)
binary_mask_img_normalized = (binary_mask_img - global_min) / (global_max - global_min)


# show normalized image
cv2.imshow('Normalized image',I_normalized)
cv2.waitKey(0)

# visualize the mask
cv2.imshow('Normalized Mask', mask_img_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# visualize the binary mask
cv2.imshow('Normalized Binary Mask', binary_mask_img_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Parameters
param = Parameters(0,0,0,0,0)
param.hi = 1 / (ni-1)
param.hj = 1 / (nj-1)

#  Parameters for gradient descent (you do not need for week1)
param.dt = 5*10^-7
param.iterMax = 10^4
param.tol = 10^-5

# print(mask_img_normalized)

# visualize the mask
cv2.imshow('Before', I)
cv2.waitKey(0)
cv2.destroyAllWindows()


I_ch1 = I_normalized[:,:, 0]
I_ch2 = I_normalized[:,:, 1]
I_ch3 = I_normalized[:,:, 2]


#-------------------------------------------------
# METHOD 1: SEAMLESS CLONING

Iinp[:,:, 0]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_seamless(I_ch1, mask_img_normalized[:, :, 0], binary_mask_img_normalized, param)
Iinp[:,:, 1]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_seamless(I_ch2, mask_img_normalized[:, :, 1], binary_mask_img_normalized, param)
Iinp[:,:, 2]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_seamless(I_ch3, mask_img_normalized[:, :, 2], binary_mask_img_normalized, param)

# visualize the final image
cv2.imshow('After, SEAMLESS CLONING', Iinp)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------


#-------------------------------------------------
# METHOD 2: MIXED SEAMLESS CLONING

Iinp[:,:, 0]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_mixed_seamless(I_ch1, mask_img_normalized[:, :, 0], binary_mask_img_normalized, param)
Iinp[:,:, 1]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_mixed_seamless(I_ch2, mask_img_normalized[:, :, 1], binary_mask_img_normalized, param)
Iinp[:,:, 2]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_mixed_seamless(I_ch3, mask_img_normalized[:, :, 2], binary_mask_img_normalized, param)

# visualize the final image
cv2.imshow('After, MIXED SEAMLESS', Iinp)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------

#-------------------------------------------------
# METHOD 3: MIXING GRADIENTS

Iinp[:,:, 0]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_mixing_gradients(I_ch1, mask_img_normalized[:, :, 0], binary_mask_img_normalized, param)
Iinp[:,:, 1]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_mixing_gradients(I_ch2, mask_img_normalized[:, :, 1], binary_mask_img_normalized, param)
Iinp[:,:, 2]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_mixing_gradients(I_ch3, mask_img_normalized[:, :, 2], binary_mask_img_normalized, param)

# visualize the final image
cv2.imshow('After, MIXING GRADIENTS', Iinp)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------


#-------------------------------------------------
# METHOD 4: SEMALESS CLONING AND DESTINATION AVERAGED

Iinp[:,:, 0]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_seamless_and_dest_averaged(I_ch1, mask_img_normalized[:, :, 0], binary_mask_img_normalized, param)
Iinp[:,:, 1]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_seamless_and_dest_averaged(I_ch2, mask_img_normalized[:, :, 1], binary_mask_img_normalized, param)
Iinp[:,:, 2]=multi_sol_Laplace_Equation_Axb.G5_sol_Laplace_Equation_Axb_python_seamless_and_dest_averaged(I_ch3, mask_img_normalized[:, :, 2], binary_mask_img_normalized, param)

# visualize the final image
cv2.imshow('After, SEAMLESS CLONING AND DESTINATION AVERAGED', Iinp)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------





#  Challenge image. (We have lost 99% of information)
del I, Iinp, mask_img