#!/home/knielbo/virtenvs/cv/bin/python
"""
Implementation of convolutions
"""
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def convolve(image, K):
    # grab spatial dimensions of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # allocate memory for the output image
    # create padding on the borders in order to keep the spatial size constant
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")
    
    # loop over input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top-to-bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract ROI by extracting the center of the region of
            # the current (x, y)-coordinates dimensions
            roi = image[y-pad : y+pad+1, x-pad : x+pad+1]

            # perform the convolution by taking the element-wise
            # multiplication between the ROI and the kernel, then
            # summing the matrix
            k = (roi * K).sum()

            # store convolved value in the output (x, y)-coordinate
            # of the output image
            output[y-pad, x-pad] = k

    # rescale output to [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

def main():
    print("yolo")


if __name__=="__main__":
    main()