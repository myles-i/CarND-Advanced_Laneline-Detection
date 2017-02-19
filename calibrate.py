import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
# Make a list of calibration images
n = 15	
nx = 9
ny = 6

# define object points
obj_p = np.zeros((nx*ny,3),	np.float32)
obj_p[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

img_points = []
obj_points = []
for i in range(1,n+1):
	name = "./camera_cal/calibration" + str(i) + ".jpg"
	img = cv2.imread(name)
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
	if ret:
		img_points.append(corners)
		obj_points.append(obj_p)

# calculate camera matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],None,None)

# save coefficients
with open('camera_coefficients.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([mtx, dist, rvecs, tvecs], f)

name = "./camera_cal/calibration1.jpg"
img = cv2.imread(name)
import pdb; pdb.set_trace()
dst = cv2.undistort(img, mtx, dist, None, mtx)

plt.figure(20,10)
plt.subplot(2,1,1)
plt.imshow(img)
plt.title('Uncorrected Image')
plt.subplot(2,1,2)
plt.imshow(dst)
plt.title('Distortion Corrected Image')