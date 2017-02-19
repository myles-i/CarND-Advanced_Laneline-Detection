import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import glob
import pickle
import pdb

# load coefficients
with open('camera_coefficients.pickle', 'rb') as f:
    mtx, dist, rvecs, tvecs = pickle.load(f)

# defined tuning parameters
sobel_min = 40
sobel_max = 80
sobel_kernel=5
x_thresh = (20,100)
y_thresh = (20,100)
abs_thresh = (20,100)
dir_thresh = (0.7, 1.2)
hls_thresh = (90, 255)

# perspectice transform parameters
buffer_pixels = 40
bottom_left_x = 244 - buffer_pixels
bottom_right_x = 1060 + buffer_pixels
top_left_x = 596 - buffer_pixels
top_right_x = 688 + buffer_pixels
bottom_y = 680
top_y    = 468
src =  np.array([[bottom_left_x,bottom_y],[top_left_x, top_y],\
                      [top_right_x, top_y],[bottom_right_x,bottom_y]],\
                      dtype=np.float32)
dst = np.array([[bottom_left_x,720],[bottom_left_x, 0],\
                      [bottom_right_x, 0],[bottom_right_x,720]],\
                      dtype=np.float32)

def warper(img, src, dst):
	# This function is a modified version of the  udacity provided solution to the warping function
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def sobel_thresh(img, sobel_kernel=3, x_thresh = (0,255), y_thresh = (0,255), abs_thresh = (0,255), dir_thresh = (0,255)):
	# This function is a modified version of the  udacity provided solution to applying the sobel operator
    # Assumes input image is in grayscale
    # Convert to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    #calculate gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output_x   = np.zeros_like(gradmag)
    binary_output_y   = np.zeros_like(gradmag)
    binary_output_mag = np.zeros_like(gradmag)
    binary_output_dir = np.zeros_like(gradmag)
    #apply thresholds
    binary_output_mag[(gradmag >= abs_thresh[0]) & (gradmag <= abs_thresh[1])] = 1
    binary_output_dir[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
    binary_output_x[(sobelx >= x_thresh[0]) & (sobelx <= x_thresh[1])] = 1
    binary_output_y[(sobely >= y_thresh[0]) & (sobely <= y_thresh[1])] = 1

    return binary_output_x, binary_output_y, binary_output_mag, binary_output_dir

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # plt.figure(1)
    # plt.imshow(mask)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    # plt.figure(5)
    # plt.imshow(masked_image)
    return masked_image

def generate_binary_image(img):
	binary_output_x, binary_output_y, binary_output_mag, binary_output_dir = \
	sobel_thresh(img, sobel_kernel=3, x_thresh = x_thresh, y_thresh = y_thresh, abs_thresh = abs_thresh, dir_thresh = dir_thresh)
	combined_grad_outout = np.zeros_like(binary_output_x)
	combined_grad_outout[((binary_output_x == 1) & (binary_output_y == 1)) | ((binary_output_mag == 1) & (binary_output_dir == 1))] = 1

	# apply hls threshold
	hls_output = hls_select(img, thresh=hls_thresh)
	binary_combined = np.zeros_like(hls_output)
	binary_combined[(hls_output == 1) | (combined_grad_outout == 1)] = 1

	return binary_combined

def preprocess(img):
	# Undistort Image
	img = cv2.undistort(img, mtx, dist, None, mtx)

	# apply thresholds to generate binary image
	binary_combined = generate_binary_image(img)

	# perspective transform
	warped_binary = warper(binary_combined, src, dst)

	return binary_combined, warped_binary

def find_lines(binary_warped, original_image, visualize = False):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	if visualize:
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		plt.figure(figsize = (5,5))
		plt.imshow(out_img)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show(block = False)



		# Create an image to draw the lines on
		warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		Minv = cv2.getPerspectiveTransform(dst,src)
		newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
		# Combine the result with the original image
		result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
		plt.figure(figsize = (5,5))
		plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
		plt.show(block = False)	

	return left_fit,right_fit



def tester():
	name = "./test_images/straight_lines1.jpg"
	name = "./test_images/test1.jpg"
	img = cv2.imread(name)

	# test binary image generator
	binary_combined, warped_binary = preprocess(img)
	plt.figure(figsize = (5,5))
	plt.imshow(binary_combined, cmap='gray')
	plt.title('Binary Image with Sobel and HLS Thresholding')
	plt.show(block = False)

	plt.figure(figsize = (5,5))
	plt.imshow(warped_binary, cmap='gray')
	plt.show(block = False)

	find_lines(warped_binary,img, visualize = True)
