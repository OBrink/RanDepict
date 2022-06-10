'''
This script has been modified from the original augment.py present on  
https://github.com/mtzgroup/ChemPixCH 

From the publication:
https://pubs.rsc.org/en/content/articlelanding/2021/SC/D1SC02957F
'''

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

'''
This file contains the augmentation functions used in the augment molecule
and augment background pipelines.
'''


def elastic_transform(image, alpha_sigma, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003. 
    https://gist.github.com/erniejunior/601cdf56d2b424757de5

    This function distords an image randomly changing the alpha and gamma
    values.

    Args:
        image: the image to modify in array format.
        alpha_sigma: alpha and sigma values randomly selected as a list. 
    Returns:
        distored_image: the image after the transformation with the same size
                        as it had originally.

    """
    alpha = alpha_sigma[0]
    sigma = alpha_sigma[1]
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=np.random.randint(1, 5), mode='reflect')
    return distored_image.reshape(image.shape)


def distort(img):
    """
    This function randomly selects a list with the shape [a, g] where 
    a=alpha and g=gamma and passes them along with the input image
    to the elastic_transform function that will do the image distorsion.

    Args:
        img: the image to modify in array format.
    Returns:
        the output from elastic_transform function which is the image 
        after the transformation with the same size as it had originally.

    """
    sigma_alpha = [(np.random.randint(9, 11), np.random.randint(2, 4)),
                   (np.random.randint(80, 100), np.random.randint(4, 5)),
                   (np.random.randint(150, 300), np.random.randint(5, 6)),
                   (np.random.randint(800, 1200), np.random.randint(8, 10)),
                   (np.random.randint(1500, 2000), np.random.randint(10, 15)),
                   (np.random.randint(5000, 8000), np.random.randint(15, 25)),
                   (np.random.randint(10000, 15000), np.random.randint(20, 25)),
                   (np.random.randint(45000, 55000), np.random.randint(30, 35))]#,
                   #(np.random.randint(350000, 400000), np.random.randint(48, 52))]
    choice = np.random.randint(len(sigma_alpha))
    sigma_alpha_chosen = sigma_alpha[choice]

    return elastic_transform(img, sigma_alpha_chosen)


def rotate(img, obj=None):
    """
    This function randomly rotates between 0-360 degrees the input 
    image. 

    Args:
        img: the image to modify in array format.
        obj: "mol" or "bkg" to modify a chemical structure image or
             a background image.
    Returns:
        dst: the rotated image.

    """
    rows,cols,_ = img.shape
    angle = np.random.randint(0,360)
    col=(float(img[0][0][0]), float(img[0][0][1]), float(img[0][0][2]))
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    if obj == "mol": 
        dst = cv2.warpAffine(img, M, (cols, rows), borderValue=[255, 255, 255])
    if obj == "bkg": 
        dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst



def resize(img):
    """
    This function resizes the image randomly from between (200-300, 200-300) 
    and then resizes it back to 256x256.

    Args:
        img: the image to modify in array format.

    Returns:
        img: the resized image.

    """
    interpolations = [cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

    img = cv2.resize(img, (np.random.randint(200, 300), np.random.randint(200, 300)), interpolation = np.random.choice(interpolations))
    img = cv2.resize(img, (256,256), interpolation = np.random.choice(interpolations))
 
    return img



def blur(img):
    """
    This function blurs the image randomly between 1-3.

    Args:
        img: the image to modify in array format.

    Returns:
        img: the blurred image.

    """
    n = np.random.randint(1,4)
    kernel = np.ones((n,n),np.float32)/n**2
    img = cv2.filter2D(img,-1,kernel)
    return img



def erode(img):
    """
    This function bolds the image randomly between 1-2.

    Args:
       img: the image to modify in array format.

    Returns:
        img: the bold image.

    """
    n=np.random.randint(1,3)
    kernel = np.ones((n,n),np.float32)/n**2
    img = cv2.erode(img, kernel, iterations=1)
    return img


def dilate(img):
    """
    This function dilates the image with a factor of 2.

    Args:
       img: the image to modify in array format.

    Returns:
        img: the dilated image.

    """
    n=2
    kernel = np.ones((n,n),np.float32)/n**2
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def aspect_ratio(img, obj=None):
    """
    This function irregularly changes the size of the image 
    and converts it back to (256,256).

    Args:
        img: the image to modify in array format.
        obj: "mol" or "bkg" to modify a chemical structure image or
             a background image.

    Returns:
        image: the resized image.

    """
    col=(float(img[0][0][0]),float(img[0][0][1]),float(img[0][0][2]))
    n1 = np.random.randint(0,50)
    n2 = np.random.randint(0,50)
    n3 = np.random.randint(0,50)
    n4 = np.random.randint(0,50)
    if obj == "mol":
        image = cv2.copyMakeBorder(img, n1, n2, n3, n4, 
                                   cv2.BORDER_CONSTANT,value=[255,255,255])
    elif obj == "bkg":
        image = cv2.copyMakeBorder(img, n1, n2, n3, n4, 
                                   cv2.BORDER_REFLECT)

    image = cv2.resize(image, (256,256))
    return image

def affine(img, obj=None):
    """
    This function randomly applies affine transformation which consists
    of matrix rotations, translations and scale operations and converts 
    it back to (256,256).

    Args:
        img: the image to modify in array format.
        obj: "mol" or "bkg" to modify a chemical structure image or
             a background image.

    Returns:
        skewed: the transformed image.

    """
    rows, cols,_ = img.shape
    n = 20
    pts1 = np.float32([[5, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[5 + np.random.randint(-n, n), 50 + np.random.randint(-n, n)],
                   [200 + np.random.randint(-n, n), 50 + np.random.randint(-n, n)],
                   [50 + np.random.randint(-n, n), 200 + np.random.randint(-n, n)]])

    M = cv2.getAffineTransform(pts1, pts2)

    if obj == "mol":
        skewed = cv2.warpAffine(img, M, (cols, rows), borderValue=[255,255,255])
    elif obj == "bkg":
        skewed = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    skewed = cv2.resize(skewed, (256,256))
    return skewed

def augment_mol(img):
    """
    This function randomly applies different image augmentations with 
    different probabilities to the input image.

    Args:
        img: the image to modify in array format.

    Returns:
        img: the augmented image.

    """
    # resize    
    if np.random.uniform(0,1) < 0.5:
        img = resize(img)
    # blur    
    if np.random.uniform(0,1) < 0.4:
        img = blur(img)
    # erode    
    if np.random.uniform(0,1) < 0.4:
        img = erode(img)
    # dilate    
    if np.random.uniform(0,1) < 0.4:
        img = dilate(img)
    # aspect_ratio    
    if np.random.uniform(0,1) < 0.7:
        img = aspect_ratio(img, "mol")
    # affine    
    if np.random.uniform(0,1) < 0.7:
        img = affine(img, "mol")
    # distort    
    if np.random.uniform(0,1) < 0.8:
        img = distort(img)
    if img.shape != (255,255,3):
        img = cv2.resize(img,(256,256))
    return img


def augment_bkg(img):
    """
    This function randomly applies different image augmentations with 
    different probabilities to the input image.

    Args:
        img: the image to modify in array format.

    Returns:
        img: the augmented image.

    """
    # rotate
    img = rotate(img, "bkg")
    # resize    
    if np.random.uniform(0,1) < 0.5:
        img = resize(img)
    # blur    
    if np.random.uniform(0,1) < 0.4:
        img = blur(img)
    # erode    
    if np.random.uniform(0,1) < 0.2:
        img = erode(img)
    # dilate    
    if np.random.uniform(0,1) < 0.2:
        img = dilate(img) 
    # aspect_ratio    
    if np.random.uniform(0,1) < 0.3:
        img = aspect_ratio(img, "bkg")
    # affine    
    if np.random.uniform(0,1) < 0.3:
        img = affine(img, "bkg")
    # distort    
    if np.random.uniform(0,1) < 0.8:
        img = distort(img)
    if img.shape != (255,255,3):
        img = cv2.resize(img,(256,256))
    return img
