import cv2
import numpy as np
from math import floor
import caffe
import sys
from matplotlib import pyplot as plt

def square_padding(ori_img):
    squared_img = ori_img
    height, width = ori_img.shape
    if (width != height):
        BLACK = [0, 0, 0]
        if (width > height):
            diff = width - height
            if (diff % 2 == 0):
                padding = int(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, padding, padding, 0, 0,
                    cv2.BORDER_CONSTANT,value=BLACK)
            else:
                padding = floor(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, padding + 1, padding, 0,0,
                    cv2.BORDER_CONSTANT,value=BLACK)
        else:
            diff = height - width
            if (diff % 2 == 0):
                padding = int(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, 0, 0, padding, padding,
                    cv2.BORDER_CONSTANT,value=BLACK)
            else:
                padding = floor(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, 0,0, padding + 1, padding,
                    cv2.BORDER_CONSTANT,value=BLACK)
    return squared_img

def get_digit_contours(thresh, contours):
    # Calculating heights and areas for all found contours
    num_contours = len(contours)
    # contour_areas = [0] * num_contours
    heights = [0] * num_contours
    img_height = thresh.shape[0]
    for i in range(0, num_contours):
        _, _, _, h = cv2.boundingRect(contours[i])
        heights[i] = h / img_height
        # contour_areas[i] = cv2.contourArea(contours[i])
    
    # Get the 9 contours which have the largest height
    desc_cnt_idx = np.flipud(np.argsort(np.array(heights)))
    num_digits = 9
    digit_contours = np.array(contours)[desc_cnt_idx[0:num_digits]]

    return num_digits, digit_contours

def validate(num_digits, digit_contours, image_area):
    box_areas = np.zeros(num_digits)
    contour_areas = np.zeros(num_digits)
    
    rects = [None] * num_digits
    for i in range(num_digits):
        x, y, w, h = cv2.boundingRect(digit_contours[i])
        rects[i] = (x, y, w, h)

    for i in range(num_digits):
        contour_areas[i] = cv2.contourArea(digit_contours[i])
        box_areas[i] = rects[i][2] * rects[i][3]

    area_ratios = box_areas / image_area
    min_ratio = np.min(area_ratios)
    max_ratio = np.max(area_ratios)
    
    if (max_ratio > 0.15 or min_ratio < 0.02):
        return False
    else:
        return True

def idToStr(img, net_prototxt, caffemodel, useGPU=True):
    # Loading image and finding countours
    # img = cv2.imread(filepath, 0)
    _, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=2)

    num_digits, digit_contours = get_digit_contours(thresh, contours)
    image_area = img.shape[0] * img.shape[1]
    is_valid = validate(num_digits, digit_contours, image_area)
    if (is_valid == False):
        kernel = np.ones((2,2),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        _, thresh = cv2.threshold(img, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=2)
        num_digits, digit_contours = get_digit_contours(thresh, contours)
        is_valid = validate(num_digits, digit_contours, image_area)
    if (is_valid == False):
        return "Cannot process this ID image, try inputing other ID image!"
    else:
        # Specify the bounding rectangle for each of selected contour
        horizontal_top_lefts = [0] * num_digits
        rects = [None] * num_digits
        for i in range(num_digits):
            x, y, w, h = cv2.boundingRect(digit_contours[i])
            rects[i] = (x, y, w, h)
            horizontal_top_lefts[i] = x

        # Get the right order of the sequence of id digits
        digit_order_idx = np.argsort(np.array(horizontal_top_lefts))

        # Save all digits into an np array
        digit_img = np.array([None] * num_digits)
        for i in range(num_digits):
            x = rects[digit_order_idx[i]][0]
            y = rects[digit_order_idx[i]][1]
            w = rects[digit_order_idx[i]][2]
            h = rects[digit_order_idx[i]][3]
            digit_img[i] = square_padding(img[y:y + h + 1, x:x + w + 1])

        # Initialise a new net
        if (useGPU == True):
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        _net = caffe.Net(net_prototxt, caffemodel, caffe.TEST)
        
        # Recognise digits individually
        id_str = ''
        for digit in digit_img:
            digit = cv2.resize(digit, (28, 28))
            # digit = digit.astype(np.float64) * (1.0 / 255.0)
            digit = digit.astype('float64') * (1.0 / 255.0)
            _net.blobs['data'].data[...] = digit[np.newaxis, np.newaxis, :, :]
            out = _net.forward()
            id_str += str(out['prob'].argmax())
        return id_str

def is_together(rect_a, rect_b, threshold_dist):
    distance_x = rect_b[0] - rect_a[0] - rect_a[2]
    distance_y_atob = rect_b[1] - rect_a[1] - rect_a[3]
    distance_y_btoa = rect_a[1] - rect_b[1] - rect_b[3]
    if (distance_x > threshold_dist
        or not (distance_y_atob <= 0 and distance_y_btoa <= threshold_dist)
        or not (distance_y_btoa <= 0 and distance_y_atob <= threshold_dist)):
        return False
    else:
        return True

def find_group(a, groups):
    is_new_group = True
    group_index = -1
    
    num_groups = len(groups)
    for i in range(num_groups):
        if a in groups[i]:
            group_index = i
            is_new_group = False
            break
    
    return is_new_group, group_index

def group_rects(rects, threshold_dist):
    rects.sort(key=lambda tup: tup[0])
    num_rects = len(rects)
    groups = []
    for i in range(num_rects):
        is_new_group, group_index = find_group(i, groups)
        if is_new_group == True:
            group = [i]
        for j in range(i + 1, num_rects):
            if is_together(rects[i], rects[j], threshold_dist):
                if is_new_group == True:
                    group.append(j)
                else:
                    groups[group_index].append(j)
        if is_new_group == True:
            groups.append(group)
    return groups

def find_longest_group(rects, groups):
    num_groups = len(groups)
    longest_group = None
    max_length = 0
    
    for i in range(num_groups):
        length = rects[groups[i][-1]][0] + rects[groups[i][-1]][2] - rects[groups[i][0]][0]
        if max_length < length:
            max_length = length
            longest_group = groups[i]
    return longest_group

def merge_rects(rects, id_groups):
    x = rects[id_groups[0]][0]
    w = rects[id_groups[-1]][0] + rects[id_groups[-1]][2]
    
    y = rects[id_groups[0]][1]
    h = rects[id_groups[0]][1] + rects[id_groups[0]][3]
    num_groups = len(id_groups)
    for i in range(1, num_groups):
        if rects[id_groups[i]][1] < y:
            y = rects[id_groups[i]][1]
        if rects[id_groups[i]][1] + rects[id_groups[i]][3] > h:
            h = rects[id_groups[i]][1] + rects[id_groups[i]][3]
    
    return x, y, w, h

def crop_id(image_path, num_top_contours=12, threshold_dist=12):
    img = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=2)
    num_contours = len(contours)
    heights = [0] * num_contours
    img_height = img.shape[0]

    for i in range(0, num_contours):
        _, _, _, h = cv2.boundingRect(contours[i])
        heights[i] = h / img_height
    desc_cnt_idx = np.flipud(np.argsort(np.array(heights)))

    rects = [None] * num_top_contours
    digit_contours = np.array(contours)[desc_cnt_idx[0:num_top_contours]]

    for i in range(num_top_contours):
        rects[i] = cv2.boundingRect(digit_contours[i])

    groups = group_rects(rects, threshold_dist)
    id_groups = find_longest_group(rects, groups)
    x, y, w, h = merge_rects(rects, id_groups)
    return img[y:h, x:w]

if __name__ == '__main__':
    id_only_img = crop_id(sys.argv[1])
    print(idToStr(id_only_img, sys.argv[2], sys.argv[3]))
    plt.imshow(id_only_img, cmap = 'gray')
    plt.show()