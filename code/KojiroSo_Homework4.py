#Import libraries and install where necessary
import numpy as np
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
import cv2
from PIL import Image
import os

#Replace with your path to where the images are stored
path = '/Users/kojiro/Downloads/homework3/html/'
#Set current working directory to provided path
os.chdir(path)
print(os.getcwd())

# create a  Binomial (5-tap) filter
kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])

def interpolate(image):
    """
    Interpolates an image with upsampling rate r=2.
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    # Upsample
    image_up[::2, ::2] = image
    # Blur (we need to scale this up since the kernel has unit area)
    # (The length and width are both doubled, so the area is quadrupled)
    #return sig.convolve2d(image_up, 4*kernel, 'same')
    return ndimage.filters.convolve(image_up,4*kernel, mode='constant')
                                
def decimate(image):
    """
    Decimates at image with downsampling rate r=2.
    """
    # Blur
    #image_blur = sig.convolve2d(image, kernel, 'same')
    image_blur = ndimage.filters.convolve(image,kernel, mode='constant')
    # Downsample
    return image_blur[::2, ::2]                                
                                                        
def pyramids(image):
    """
    Constructs Gaussian and Laplacian pyramids.
    Parameters :
        image  : the original image (i.e. base of the pyramid)
    Returns :
        G   : the Gaussian pyramid
        L   : the Laplacian pyramid
    """
    # Initialize pyramids
    G = [image, ]
    L = []

    # Build the Gaussian pyramid to maximum depth
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)
        
    # Build the Laplacian pyramid
    for i in range(len(G) - 1):
        L.append(G[i] - interpolate(G[i + 1]))

    return G[:-1], L

def reconstruct(L,G):
  rows, cols = img.shape
  composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
  composite_image[:rows, :cols] = G[0]

  i_row = 0
  for p in G[1:]:
      n_rows, n_cols = p.shape[:2]
      composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
      i_row += n_rows


  fig, ax = plt.subplots()
      
  ax.imshow(composite_image,cmap='gray')
  plt.show()


  rows, cols = img.shape
  composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)

  composite_image[:rows, :cols] = L[0]

  i_row = 0
  for p in L[1:]:
      n_rows, n_cols = p.shape[:2]
      composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
      i_row += n_rows


  fig, ax = plt.subplots()
    
  ax.imshow(composite_image,cmap='gray')
  plt.show()

def image_prep(image1, image2, mask):
  image1 = imageio.imread(path + image1, as_gray = True)
  image2 = imageio.imread(path + image2, as_gray = True)
  mask = imageio.imread(path + 'mask.jpg',as_gray = True)
  image1 = cv2.resize(image1, (512, 512))
  image2 = cv2.resize(image2, (512, 512))
  return image1, image2, mask

def laplacian_pyramid(image, bools = False):
  laplacian_pyramid = pyramids(image)[1]
  fig, ax = plt.subplots(1, 9, figsize = (20,20))
  for i in range(len(laplacian_pyramid)):
    ax[i].imshow(laplacian_pyramid[i], cmap = 'gray')
  i = 0
  while os.path.exists('{}{:d}.jpg'.format('laplacian_pyramid', i)):
    i += 1
  if bools == True:    
    fig.savefig('{}{:d}.jpg'.format('laplacian_pyramid', i), bbox_inches='tight')
  plt.close()  
  return laplacian_pyramid

def gaussian_pyramid(image, bools = False):
  gaussian_pyramid = pyramids(image)[0]
  fig, ax = plt.subplots(1, 9, figsize = (20,20))
  for i in range(len(gaussian_pyramid)):
    ax[i].imshow(gaussian_pyramid[i], cmap = 'gray')
  i = 0
  while os.path.exists('{}{:d}.jpg'.format('gaussian_pyramid', i)):
    i += 1
  if bools == True:    
    fig.savefig('{}{:d}.jpg'.format('gaussian_pyramid', i), bbox_inches='tight')
  plt.close()  
  return gaussian_pyramid

def gaussian_mask(mask, bools = False):
  gaussian_pyramid_mask = pyramids(mask)[0]
  fig, ax = plt.subplots(1, 9, figsize = (20,20))
  for i in range(len(gaussian_pyramid_mask)):
    ax[i].imshow(gaussian_pyramid_mask[i], cmap = 'gray')
  if bools == True:  
    fig.savefig('gaussian_pyramid_mask.jpg', bbox_inches='tight')
  plt.close()  
  return gaussian_pyramid_mask

def gaussian_mask_inverted(mask, bools = False):
  flipped = np.flip(mask)
  gaussian_pyramid_mask_inverted = pyramids(flipped)[0]
  fig, ax = plt.subplots(1, 9, figsize = (20,20))
  for i in range(len(gaussian_pyramid_mask_inverted)):
    ax[i].imshow(gaussian_pyramid_mask_inverted[i], cmap = 'gray')  
  if bools == True:
    fig.savefig('gaussian_pyramid_mask_inverted.jpg', bbox_inches='tight')
  plt.close()  
  return gaussian_pyramid_mask_inverted

def merge_laplacians(laplacian_pyramid_image1, laplacian_pyramid_image2, gaussian_mask_image1, gaussian_mask_image2):
  #Combine each laplacian level (masked) of each pyramid 
  #(i.e. each level of laplacian_pyramid_orange + laplacian_pyramid_apple)
  combined_laplacian = []
  for i in range(-2, -len(laplacian_pyramid_image1)-1, -1):
    lap_image1 = laplacian_pyramid_image1[i] * gaussian_mask_image1[i]
    lap_image2 = laplacian_pyramid_image2[i] * gaussian_mask_image2[i]
    combined_laplacian.append(lap_image1 + lap_image2)  
  return combined_laplacian

def merge_gaussians(gaussian_pyramid_image1, gaussian_pyramid_image2, gaussian_mask_image1, gaussian_mask_image2):
  #Combine lowest gaussian level (masked) of each pyramid
  gaus_image1 = interpolate(gaussian_pyramid_image1[-1]) * gaussian_mask_image1[-2]
  gaus_image2 = interpolate(gaussian_pyramid_image2[-1]) * gaussian_mask_image2[-2]
  combined_gaussian = gaus_image1 + gaus_image2
  return combined_gaussian

def blend(merged_laplacian_result, merged_gaussian_result, bools = False):
  #Blend orange and apple together
  blended = merged_laplacian_result[0] + merged_gaussian_result
  blended_final = blended
  for i in range(1, 8):
    blended_final = interpolate(blended_final) + merged_laplacian_result[i]
  fig, ax = plt.subplots(1,1, figsize = (5,5))
  ax.imshow(blended_final, cmap = 'gray')
  i = 0
  while os.path.exists('{}{:d}.jpg'.format('blended_final', i)):
    i += 1
  if bools == True:    
    fig.savefig('{}{:d}.jpg'.format('blended_final', i), bbox_inches='tight')
  plt.close()  
  return blended_final

def reconstruct_channels(blended, rotate = False):
	#Reconstruct channels and stack in color channels
  stacked_img = np.stack((blended, )*3, axis = -1)
  stacked_img = stacked_img/np.amax(stacked_img)
  if rotate == True:
    stacked_img = np.rot90(stacked_img, k = 3)
  fig, ax = plt.subplots(1,1, figsize = (5,5))
  ax.imshow(stacked_img, cmap ='gray')
  i = 0
  while os.path.exists('{}{:d}.jpg'.format('final_image', i)):
    i += 1
  fig.savefig('{}{:d}.jpg'.format('final_image', i), bbox_inches='tight')
  plt.close()
  return stacked_img

def blend_images(image1, image2, mask, bools = False, rotate = False):
  image1, image2, mask = image_prep(image1, image2, mask)
  laplacian_image1 = laplacian_pyramid(image1, bools)
  laplacian_image2 = laplacian_pyramid(image2, bools)
  gaussian_image1 = gaussian_pyramid(image1, bools)
  gaussian_image2 = gaussian_pyramid(image2, bools)
  image1_mask = gaussian_mask(mask, bools)
  image2_mask = gaussian_mask_inverted(mask, bools)
  merged_laplacian = merge_laplacians(laplacian_image1, laplacian_image2, image1_mask, image2_mask)
  merged_gaussian = merge_gaussians(gaussian_image1, gaussian_image2, image1_mask, image2_mask)
  blended_image = blend(merged_laplacian, merged_gaussian, bools)
  final_image = reconstruct_channels(blended_image, rotate)

apple_orange = blend_images('orange.jpg', 'apple.jpg','mask.jpg', bools = True)
centaur = blend_images('man_crawling.png', 'horse_up.png', 'mask.jpg')
dog_portrait = blend_images('Picture2.jpg', 'Picture1.jpg', 'mask.jpg')
sun_flower = blend_images('sun.png', 'flower.png', 'mask.jpg')
falling_woman = blend_images('water.png', 'falling2.png', 'mask.jpg', rotate = True)



