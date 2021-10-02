import matplotlib
import numpy as np
import random
import time
import cv2
import scipy
from PIL import Image


# **********************************************************************************
# --------- 2 ---------------
# **********************************************************************************
def compute_homography_naive(mp_src, mp_dst):
    A = np.zeros((mp_src.shape[1] * 2, 9))
    coord_src = np.concatenate((mp_src.T, np.ones((mp_src.shape[1], 1))), axis=1)
    coord_dst = np.concatenate((mp_dst.T, np.ones((mp_dst.shape[1], 1))), axis=1)

    # build matrix A
    for i in range(0, mp_src.shape[1]):
        A[2 * i, 0:3] = coord_src[i]
        A[2 * i, 6:9] = coord_src[i] * coord_dst[i, 0]*(-1)

    for i in range(0, mp_src.shape[1]):
        A[2 * i + 1, 3:6] = coord_src[i]
        A[2 * i + 1, 6:9] = coord_src[i] * coord_dst[i, 1]*(-1)

    # calculate eigenvalues
    AA = np.matmul(A.T, A)
    w, v = np.linalg.eig(AA)
    h = v[:, w == min(w)]
    if h.shape[1] != 1:
        h = h[:, 0]
    H = h.reshape(3, 3)
    # normalize H
    H = H/H[2, 2]
    return H


# **********************************************************************************
# --------- Forward Mapping ---------------
# **********************************************************************************
def get_grid(x, y, homogenous=False):
    # get all possible coordinates as column vectors
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords


def forward_mapping(image, A, get_bounding_rect=False):
    # Points generator
    # Grid to represent image coordinate
    height, width = image.shape[:2]

    if get_bounding_rect:
        corners = [[0, width-1, 0, width-1], [0, 0, height-1, height-1], [1, 1, 1, 1]]
        warp_corners = A @ corners
        x_warp_cor, y_warp_cor = warp_corners[0, :] / warp_corners[2, :], warp_corners[1, :] / warp_corners[2, :]
        return x_warp_cor, y_warp_cor

    coords = get_grid(width, height, True)
    x_ori, y_ori = coords[0], coords[1]

    # Apply transformation
    warp_coords = A@coords
    x_warp, y_warp = warp_coords[0, :]/warp_coords[2,:], warp_coords[1, :]/warp_coords[2,:]

    # translate pixel coordinates so it's >= 0
    x_warp = x_warp - min(x_warp)
    y_warp = y_warp - min(y_warp)

    # cast to int
    x_warp = x_warp.astype(int)
    y_warp = y_warp.astype(int)
    x_ori = x_ori.astype(int)
    y_ori = y_ori.astype(int)

    # Get pixels within image boundaries - ignore points where the denominator is too small
    indices = np.where(warp_coords[2,:] > 0.1)

    xpix_ori, ypix_ori = x_ori[indices], y_ori[indices]
    # xpix_warp, ypix_warp = np.ravel(warp_coords[0,indices]), np.ravel(warp_coords[1,indices])
    xpix_warp, ypix_warp = np.ravel(warp_coords[0, indices]/warp_coords[2,indices]), np.ravel(warp_coords[1, indices]/warp_coords[2,indices])

    # cast to int
    xpix_warp = xpix_warp.astype(int)
    ypix_warp = ypix_warp.astype(int)

    if xpix_warp.size < x_warp.size :
        # that is for the case when H is wrong and the image is too big
        canvas = np.zeros([max(ypix_warp) + 1, max(xpix_warp) + 1, 3], dtype=np.uint8)
        canvas[ypix_warp, xpix_warp] = image[ypix_ori, xpix_ori]
    else :
        canvas = np.zeros([max(y_warp)+1, max(x_warp)+1, 3], dtype=np.uint8)
        canvas[y_warp, x_warp] = image[y_ori, x_ori]

    return canvas


# **********************************************************************************
# ---------  7  ---------------
# **********************************************************************************
def test_homography(H, mp_src, mp_dst, max_err, give_inliers=False):
    src_coord = np.vstack((mp_src, np.ones(mp_src.shape[1])))

    warp_coords = H@src_coord
    warp_coords = warp_coords/warp_coords[2,:]
    warp_coords = warp_coords[0:2,:]
    errors = np.sqrt(np.square(mp_dst[0,:] - warp_coords[0,:])+np.square(mp_dst[1,:] - warp_coords[1,:]))

    # find inliers
    inliers_src = mp_src[:, errors <= max_err]
    inliers_dest = mp_dst[:, errors <= max_err]
    inliers_err = errors[errors <= max_err]
    fit_per = inliers_src.size / mp_src.size
    dist_mse = np.mean(np.square(inliers_err))

    if give_inliers:
        return fit_per, dist_mse, inliers_src, inliers_dest

    return fit_per, dist_mse


# **********************************************************************************
# ---------  8  ---------------
# **********************************************************************************
def compute_homography(match_p_src, match_p_dst, inliers_percent, max_err):
    num_of_iteration = np.ceil(np.log(1-0.9999)/np.log(1-np.power(inliers_percent, 4))).astype(int)
    # demand certainty of 99.99%

    best_H = None
    best_err = 9999999
    for i in range(num_of_iteration):
        list_of_pair = list(zip(match_p_src.transpose(), match_p_dst.transpose()))
        samples = random.sample(list_of_pair, 4)
        src_4, dst_4 = zip(*samples)
        src_4 = np.asarray(src_4).transpose()
        dst_4 = np.asarray(dst_4).transpose()

        current_H = compute_homography_naive(src_4, dst_4)
        fit_per, dist_mse, inliers_src, inliers_dst = test_homography(current_H, match_p_src, match_p_dst, max_err, give_inliers=True)
        if fit_per >= 0.3:
            current_H = compute_homography_naive(inliers_src, inliers_dst)
            fit_per, dist_mse = test_homography(current_H, match_p_src, match_p_dst, max_err)
            if dist_mse < best_err:
                best_err = dist_mse
                best_H = current_H;
#                 # print(f'*******Best model: fit percent = {fit_per}% , MSE = {Best_err} *********')

        # print(f'found {fit_per}% in iter {i}')

    return best_H


# **********************************************************************************
# ---------  11  ---------------
# **********************************************************************************
def get_mapped_corners(img, H):
    x_corner_dst, y_corner_dst = forward_mapping(img, H, get_bounding_rect=True)

    # build rectangle
    corners = list(zip(x_corner_dst, y_corner_dst))
    # sort w.r.t x from min to max
    corners.sort()
    x_corner_dst, y_corner_dst = zip(*corners)
    x_corner_dst, y_corner_dst = np.asarray(x_corner_dst), np.asarray(y_corner_dst)
    if y_corner_dst[0] < y_corner_dst[1]:
        corner_top_left = np.array([[x_corner_dst[0]], [y_corner_dst[0]]]).astype(int)
        corner_bottom_left = np.array([[x_corner_dst[1]], [y_corner_dst[1]]]).astype(int)
    else :
        corner_top_left = np.array([[x_corner_dst[1]], [y_corner_dst[1]]]).astype(int)
        corner_bottom_left = np.array([[x_corner_dst[0]], [y_corner_dst[0]]]).astype(int)

    if y_corner_dst[2] < y_corner_dst[3]:
        corner_top_right = np.array([[x_corner_dst[2]], [y_corner_dst[2]]]).astype(int)
        corner_bottom_right = np.array([[x_corner_dst[3]], [y_corner_dst[3]]]).astype(int)
    else :
        corner_top_right = np.array([[x_corner_dst[3]], [y_corner_dst[3]]]).astype(int)
        corner_bottom_right = np.array([[x_corner_dst[2]], [y_corner_dst[2]]]).astype(int)

    bound_x_min = int(min(x_corner_dst))
    bound_x_max = int(max(x_corner_dst))
    bound_y_min = int(min(y_corner_dst))
    bound_y_max = int(max(y_corner_dst))

    corners = [corner_top_left, corner_bottom_left, corner_top_right, corner_bottom_right]

    return bound_x_min, bound_x_max, bound_y_min, bound_y_max, corners


# **********************************************************************************
# ---------  12  ---------------
# **********************************************************************************
def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):

    # find homography matrix using RANSAC
    H_ransac = compute_homography(mp_src, mp_dst, inliers_percent, max_err)

    # make sure the H_ransac is good fit
    fit_per, dist_mse = test_homography(H_ransac, mp_src, mp_dst, max_err)
    # print(f'H test : fit percent = {fit_per}%   MSE = {dist_mse}')

    # find the mapped src image corners and rect size
    src_rect_x_min, src_rect_x_max, src_rect_y_min, src_rect_y_max, corners = get_mapped_corners(img_src, H_ransac)
    src_rect_width = src_rect_x_max - src_rect_x_min
    src_rect_height = src_rect_y_max - src_rect_y_min

    # dst image rect size
    dst_rect_width = img_dst.shape[1]
    dst_rect_height = img_dst.shape[0]

    # compute translation with respect to the x bound and y bound to show the entire mapped image
    # translate each mapped src axis only if it is mapped to negative coordinates
    t = [max(-src_rect_x_min, 0), max(-src_rect_y_min, 0)]
    H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # compute the full panorama rect size
    pano_width = max(src_rect_x_max, dst_rect_width) - min(src_rect_x_min, 0)
    pano_height = max(src_rect_y_max, dst_rect_height) - min(src_rect_y_min, 0)

    # perform backward mapping, choose bi-linear interpolation with flags = 1
    # add the translation by matrix multiplication from the left so the whole image is shown
    img_pan = cv2.warpPerspective(img_src, H_t@H_ransac, (pano_width, pano_height), flags=1, borderMode=0, borderValue=0)

    # add dst image to the panorama, in overlapping areas take pixel value of dst
    # find top left corner of dst image in the panorama image
    dst_x_min = max(t[0], 0) # if t[0] is negative choose 0
    dst_y_min = max(t[1], 0) # if t[1] is negative choose 0

    img_pan[dst_y_min:dst_rect_height+dst_y_min, dst_x_min:dst_rect_width+dst_x_min] = img_dst[:, :]

    return img_pan
