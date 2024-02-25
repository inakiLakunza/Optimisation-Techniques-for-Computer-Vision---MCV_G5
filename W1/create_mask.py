import os

import cv2
import numpy as np

#img = cv2.imread("./../inpaintings/image12_toRestore.png", cv2.IMREAD_UNCHANGED)

img = cv2.imread("./image_to_Restore.png", cv2.IMREAD_UNCHANGED)
mask = img.copy() 
mask = ((mask[:, :, 2] == 255) & (mask[:, :, 1] == 0)  & (mask[:, :, 0] == 0)) * 255
mask=mask.astype('float')

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.imshow('mask', mask)
cv2.waitKey(0)

cv2.imwrite("./image_to_Restore_mask.jpg", mask)