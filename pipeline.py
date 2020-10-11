import cv2, os
import numpy as np
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
import math

#============================s

# Kích thước ảnh cho input của model
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def grayscale(img):
     return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
     return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def isolate_color_mask(img, low_thresh, high_thresh):
     assert(low_thresh.all() >=0  and low_thresh.all() <=255)
     assert(high_thresh.all() >=0 and high_thresh.all() <=255)
     return cv2.inRange(img, low_thresh, high_thresh)

# Darkened the grayscale image
def adjust_gamma(image, gamma=1.0):
     invGamma = 1.0 / gamma
     table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")

     # apply gamma correction using the lookup table
     return cv2.LUT(image, table)

def  crop(image):
     """
     Cắt bỏ bầu trời và mũi xe trong ảnh
     """
     return image[100:175, :, :]


def  resize(image):
     """
     Resize ảnh
     """
     return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

#================================================

#==================================================
def  abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	"""
	Takes an image, gradient orientation, and threshold min/max values
	"""
	# Convert to grayscale
     #img_demo = cv2.GaussianBlur(img, (3, 3), 0)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Return the result
	return binary_output

def  mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


def  dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	Return the direction of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output


def  hls_thresh(img, thresh=(100, 255)):
	"""
	Convert RGB to HLS and threshold to binary image using S channel
	"""
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output

def combined_thresh(image):
     abs_bin = abs_sobel_thresh(image, orient='x', thresh_min=50, thresh_max=255)
     mag_bin = mag_thresh(image, sobel_kernel=3, mag_thresh=(50, 255))
     dir_bin = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
     hls_bin = hls_thresh(image, thresh=(170, 255)) #default = 170
     combined = np.zeros_like(dir_bin)
     combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1
     return combined, abs_bin, mag_bin, dir_bin, hls_bin  # DEBUG

#===========================
# set point

def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]

    src = np.float32([[w - 4, h],    # br
                      [0, h],    # bl
                      [64, 10],   # tl
                      [125, 10]])  # tr
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    #unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

    return warped, M, Minv

#============================
"""
demo thuật toán 

# """
# def tim_AvaB(image, tam_xe, vitridau, vitricuoi):
     


# def return_steering(image):
#      h, w = image.shape[:2]
#      # h = 66, w = 220
#      # tam cua xe se la w / 2
#      """
#      A*         *B
#           *O
#           H la tam cua AB
#           E la tam duong cao cua tam giac AOB
#           khoang cach giua 2 duong dut cua xe toi da 28 toi thieu la 10
#           => 18
#           lay 1/2 cua anh de quet la vua
#      """
#      tam_xe = w / 2

#      # target : tim 2 diem A va B



#============================



def  pipeline_image(image):
     # step 1
     image = crop(image)
     # step 2
     image = resize(image)

     image1 = image
     # step 3
     # # mutil process image
     """
     4 mask to process image
     """
     combined, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(image)
     #warped, M, Minv = birdeye(combined, False)

     #=============================
     """
     demo làm tối 1 góc của ảnh
     """

     #==================

     thresh = cv2.threshold(combined, 0, 5, cv2.THRESH_BINARY)[1]
     gaussian_img = gaussian_blur(thresh, kernel_size = 5)
     slice1Copy = np.uint8(gaussian_img)
     img_copy = combined

     img_canny = cv2.Canny(slice1Copy, 0, 5)
     edged = np.repeat(img_canny[..., np.newaxis], 3, -1)

     return edged, img_canny