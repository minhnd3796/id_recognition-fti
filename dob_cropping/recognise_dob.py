import cv2
import numpy as np

def crop_dob(img):
    # Get contours
    _, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=2)
    
    # Initialise data structures
    num_contours = len(contours)
    contour_areas = [0] * num_contours
    heights = [0] * num_contours
    
    # Get heights for all contours
    for i in range(num_contours):
        _, _, _, h = cv2.boundingRect(contours[i])
        heights[i] = h
    
    # Descendingly sort heights of contours
    desc_cnt_idx = np.flipud(np.argsort(np.array(heights)))
    
    # Retain the 8 highest contours
    if num_contours < 8:
        max_num_contours = num_contours
    else:
        max_num_contours = 8
    horizontal_top_lefts = [0] * max_num_contours
    rects = [None] * max_num_contours
    digit_contours = np.array(contours)[desc_cnt_idx[0:max_num_contours]]
    
    # Get rects for all retained contours and get x-coordinate values for all of them
    for i in range(max_num_contours):
        rects[i] = cv2.boundingRect(digit_contours[i])
        horizontal_top_lefts[i] = rects[i][0]
        
    # Ascendingly sort x-coordinate values
    asc_countour_idx = np.argsort(np.array(horizontal_top_lefts))
    
    # Get extreme points for the desired cropping area
    rect_nparray = np.array(rects)
    start_col = rect_nparray[asc_countour_idx][0, 0]
    end_col = rect_nparray[asc_countour_idx][max_num_contours - 1, 0] + rect_nparray[asc_countour_idx][max_num_contours - 1, 2]
    start_row = np.min(rect_nparray[:, 1])
    end_row = np.max(rect_nparray[:, 1] + rect_nparray[:, 3])
    
    # Crop the input image
    cropped_img = img[start_row:end_row, start_col:end_col]
    img_w = cropped_img.shape[1]
    
    # Rough ratio for each region (day, month, year)
    dd_end = 0.22
    mm_start = 0.29
    mm_end = 0.51
    yyyy_start = 0.56
    
    # Crop day, month and year region (roughly)
    dd_img = cropped_img[:, :int(img_w * dd_end)]
    mm_img = cropped_img[:, int(img_w * mm_start):int(img_w * mm_end)]
    yyyy_img = cropped_img[:, int(img_w * yyyy_start):]
    return dd_img, mm_img, yyyy_img