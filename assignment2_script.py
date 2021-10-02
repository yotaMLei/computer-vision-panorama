import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import time

import numpy as np
import cv2

from ex2_functions import *
def tic():
    return time.time()
def toc(t):
    return float(tic()) - float(t)

##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = 204095632
ID2 = 300037397
##########################################################


# Parameters
max_err = 25
inliers_percent = 0.8

# Read the data:
img_src = mpimg.imread('src.jpg')
img_dst = mpimg.imread('dst.jpg')
# matches = scipy.io.loadmat('matches') #matching points and some outliers
matches = scipy.io.loadmat('matches_perfect') #loading perfect matches
match_p_dst = matches['match_p_dst'].astype(float)
match_p_src = matches['match_p_src'].astype(float)

matches_outliers = scipy.io.loadmat('matches') #loading matches
match_p_dst_outliers = matches_outliers['match_p_dst'].astype(float)
match_p_src_outliers = matches_outliers['match_p_src'].astype(float)


# **********************************************************************************
# ---------  Preparatory Steps ---------------
# **********************************************************************************
plt.subplot(221)
plt.imshow(img_src)
plt.scatter(match_p_src[0], match_p_src[1], c='red')
plt.title("source image perfect match points")

plt.subplot(222)
plt.imshow(img_dst)
plt.scatter(match_p_dst[0], match_p_dst[1], c='red')
plt.title("dest image perfect match points")

plt.subplot(223)
plt.imshow(img_src)
plt.scatter(match_p_src_outliers[0], match_p_src_outliers[1], c='red')
plt.title("source image match points with outliers")

plt.subplot(224)
plt.imshow(img_dst)
plt.scatter(match_p_dst_outliers[0], match_p_dst_outliers[1], c='red')
plt.title("dest image match points with outliers")

plt.show()


# **********************************************************************************
# ---------  3  ---------------
# **********************************************************************************
# Compute naive homography
tt = time.time()
H_naive = compute_homography_naive(match_p_src, match_p_dst)
print('Naive Homography {:5.4f} sec'.format(toc(tt)))
print(H_naive)


# # **********************************************************************************
# # ---------  4  ---------------
# # **********************************************************************************
# forward mapping
f_img = forward_mapping(img_src, H_naive)
plt.imshow(f_img)
plt.title("source image forward mapping with perfect match points")
plt.show()



# # **********************************************************************************
# # ---------  6  ---------------
# # **********************************************************************************
# Compute naive homography
tt = time.time()
H_naive_out = compute_homography_naive(match_p_src_outliers, match_p_dst_outliers)
print('Naive Homography {:5.4f} sec'.format(toc(tt)))
print(H_naive_out)

# forward mapping
f_img_out = forward_mapping(img_src, H_naive_out)
plt.imshow(f_img_out)
plt.title("source image forward mapping with outliers naive")
plt.show()


# **********************************************************************************
# ---------  7  ---------------
# **********************************************************************************
# Test naive homography
tt = time.time()
fit_percent, dist_mse = test_homography(H_naive, match_p_src, match_p_dst, max_err)
print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

# Test naive homography outliers
tt = time.time()
fit_percent, dist_mse = test_homography(H_naive_out, match_p_src_outliers, match_p_dst_outliers, max_err)
print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

# **********************************************************************************
# ---------  10  ---------------
# **********************************************************************************
# Compute RANSAC homography
tt = tic()
H_ransac = compute_homography(match_p_src_outliers, match_p_dst_outliers, inliers_percent, max_err)
print('RANSAC Homography {:5.4f} sec'.format(toc(tt)))
print(H_ransac)

# Test RANSAC homography
tt = tic()
fit_percent, dist_mse = test_homography(H_ransac, match_p_src_outliers, match_p_dst_outliers, max_err)
print('RANSAC Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

# forward mapping
f_img_out = forward_mapping(img_src, H_ransac)
plt.imshow(f_img_out)
plt.title("source image forward mapping with outliers using RANSAC")
plt.show()


# **********************************************************************************
# ---------  11  ---------------
# **********************************************************************************
# backward mapping
# forward map the corners of src_img to get bounding rectangle
bound_x_min, bound_x_max, bound_y_min, bound_y_max, corners = get_mapped_corners(img_src, H_ransac)
rect_width = bound_x_max - bound_x_min
rect_height = bound_y_max - bound_y_min

# add translation with respect to the x bound and y bound to show the entire image
t = [-bound_x_min, -bound_y_min]
H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

# perform backward mapping, choose bi-linear interpolation with flags = 1
# add the translation by matrix multiplication from the left
b_img_out = cv2.warpPerspective(img_src, H_t@H_ransac, (rect_width, rect_height), flags=1, borderMode=0, borderValue=0)
plt.imshow(b_img_out)
plt.title("source image backward mapping with outliers using RANSAC")
plt.show()


# **********************************************************************************
# ---------  13  ---------------
# **********************************************************************************
# Build panorama
tt = tic()
img_pan = panorama(img_src, img_dst, match_p_src_outliers, match_p_dst_outliers, inliers_percent, max_err)
print('Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Great Panorama')
plt.show()


# **********************************************************************************
# ---------  14  ---------------
# **********************************************************************************
## Student Files
#first run "create_matching_points.py" with your own images to create a mat file with the matching coordinates.
max_err = 25 # <<<<< YOU MAY CHANGE THIS
inliers_percent = 0.5 # <<<<< YOU MAY CHANGE THIS

img_src_test = mpimg.imread('src_test.jpg')
img_dst_test = mpimg.imread('dst_test.jpg')

matches_test = scipy.io.loadmat('matches_test')

match_p_dst = matches_test['match_p_dst']
match_p_src = matches_test['match_p_src']

print(match_p_dst.shape, match_p_src.shape)

# Build student panorama

tt = tic()
img_pan = panorama(img_src_test, img_dst_test, match_p_src, match_p_dst, inliers_percent, max_err)
print('Student Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Awesome Panorama')
plt.show()


plt.subplot(1,2,1)
plt.imshow(img_src_test)
plt.scatter(match_p_src[0], match_p_src[1], c='red')
plt.title("student image 1 match points")

plt.subplot(1,2,2)
plt.imshow(img_dst_test)
plt.scatter(match_p_dst[0], match_p_dst[1], c='red')
plt.title("student image 2 match points")
plt.show()
