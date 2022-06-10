'''
This script has been modified from the original degrade.py present on  
https://github.com/mtzgroup/ChemPixCH 

From the publication:
https://pubs.rsc.org/en/content/articlelanding/2021/SC/D1SC02957F
'''

import cv2
import numpy as np
from PIL import Image, ImageEnhance

def contrast(img):
    """
    This function randomly changes the input image contrast.
    
    Args:
        img: the image to modify in array format.
    Returns:
        img: the image with the contrast changes.
    """
    if np.random.uniform(0, 1)<0.8: # increase contrast
        f = np.random.uniform(1, 2)
    else: # decrease contrast
        f = np.random.uniform(0.5, 1)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(im_pil)
    im  = enhancer.enhance(f)
    img = np.asarray(im)
    return np.asarray(im)

def brightness(img):
    """
    This function randomly changes the input image brightness.
    
    Args:
        img: the image to modify in array format.
    Returns:
        img: the image with the brightness changes.
    """
    f = np.random.uniform(0.4, 1.1)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(im_pil)
    im  = enhancer.enhance(f)
    img = np.asarray(im)
    return np.asarray(im)

def sharpness(img):
    """
    This function randomly changes the input image sharpness.
    
    Args:
        img: the image to modify in array format.
    Returns:
        img: the image with the sharpness changes.
    """
    if np.random.uniform(0,1) < 0.5: # increase sharpness
        f = np.random.uniform(0.1, 1)
    else: # decrease sharpness
        f = np.random.uniform(1, 10)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(im_pil)
    im  = enhancer.enhance(f)
    img = np.asarray(im)
    return np.asarray(im)

def s_and_p(img):
    """
    This function randomly adds salt and pepper to the input image.
    
    Args:
        img: the image to modify in array format.
    Returns:
        out: the image with the s&p changes.
    """
    amount = np.random.uniform(0.001, 0.01)
    # add some s&p
    row, col = img.shape[:2]
    s_vs_p = 0.5
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in img.shape]
    out[tuple(coords)] = 1
    #pepper
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in img.shape]
    out[tuple(coords)] = 0
    return out

def scale(img):
    """
    This function randomly scales the input image.
    
    Args:
        img: the image to modify in array format.
    Returns:
        res: the scaled image.
    """
    f = np.random.uniform(0.5, 1.5)
    shape_OG = img.shape
    res = cv2.resize(img, None, fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
    res = cv2.resize(res, None, fx=1.0/f, fy=1.0/f, interpolation = cv2.INTER_CUBIC)
    return res

def degrade_img(img):
    """
    This function randomly degrades the input image by applying different
    degradation steps with different robabilities.

    Args:
        img: the image to modify in array format.
    Returns:
        img: the degraded image.
    """
    # s+p    
    if np.random.uniform(0, 1) < 0.1:
        img = s_and_p(img)

    # scale
    if np.random.uniform(0, 1) < 0.5:
        img = scale(img)

    # brightness
    if np.random.uniform(0 ,1) < 0.7:
        img = brightness(img)        

    # contrast
    if np.random.uniform(0, 1) < 0.7:
        img = contrast(img)

    # sharpness
    if np.random.uniform(0, 1) < 0.5:
        img = sharpness(img)

    #Modify the next line if you want a particular image size as output
    #img = cv2.resize(img, (256, 256))
    return img
