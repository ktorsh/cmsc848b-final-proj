import numpy as np
import cv2

def addNoise(image, mean = 0,  sigma = 0.3):
  ''' 
  This function takes an image and returns an image that has been noised with the given input parameters.
  p - Probability threshold of salt and pepper noise.
  noisetype - 
  '''
  sigma *= 255 #Since the image itself is not normalized
  noise = np.zeros_like(image)
  noise = cv2.randn(noise, mean, sigma)
  ret = cv2.add(image, noise) #generate and add gaussian noise
  return ret
