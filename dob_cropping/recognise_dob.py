import cv2
import numpy as np
from recognise_id_number import square_padding
import caffe
from matplotlib import pyplot as plt
from sys import argv

def crop_dob(img):
    # Get contours
    _, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=2)
    
    # Initialise data structures
    num_contours = len(contours)
    # contour_areas = [0] * num_contours
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

def numberSeq2Str(digit_img_arr, net):
    id_str = ''
    for digit in digit_img_arr:
        digit = cv2.resize(digit, (28, 28))
        digit = digit.astype(np.float64) * (1.0 / 255.0)
        net.blobs['data'].data[...] = digit[np.newaxis, np.newaxis, :, :]
        out = net.forward()
        id_str += str(out['prob'].argmax())
    return id_str

def recognise_dd_mm_yyyy(img, img_type, net):
    assert img_type == 'dd' or img_type == 'mm' or img_type == 'yyyy'

    if img_type == 'dd' or img_type == 'mm':
        num_digits = 2
    elif img_type == 'yyyy':
        num_digits = 4

    _, w = img.shape
    digit_img = [None] * num_digits
    chunk_w = int(w / num_digits)
    for i in range(num_digits):
        start_pixel = i * chunk_w
        digit_img[i] = img[:, start_pixel:start_pixel + chunk_w]
    
    for i in range(len(digit_img)):
        _, thresh = cv2.threshold(digit_img[i], 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=2)

        num_contours = len(contours)
        rect = [None] * num_contours
        contour_areas = [0.] * num_contours

        for j in range(num_contours):
            x, y, w, h = cv2.boundingRect(contours[j])
            rect[j] = (x, y, w, h)
            contour_areas[j] = w * h

        desc_cnt_idx = np.flipud(np.argsort(np.array(contour_areas)))
        (x, y, w, h) = rect[desc_cnt_idx[0]]
        digit_img[i] = square_padding(digit_img[i][y:y + h, x:x + w])
        plt.imshow(digit_img[i], cmap='gray')
        plt.show()
    return int(numberSeq2Str(digit_img, net))

def recognise_dob(file_path, use_gpu=True):
    assert use_gpu == True or use_gpu == False
    dob_img = cv2.imread(file_path, 0)
    plt.imshow(dob_img, cmap='gray')
    plt.show()
    dd_img, mm_img, yyyy_img = crop_dob(dob_img)
    if use_gpu == True:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    _net = caffe.Net('../lenet_deploy.prototxt', '../lenet_iter_100000.caffemodel', caffe.TEST)
    
    plt.imshow(dd_img, cmap='gray')
    plt.show()
    d = recognise_dd_mm_yyyy(dd_img, 'dd', _net)

    plt.imshow(mm_img, cmap='gray')
    plt.show()
    m = recognise_dd_mm_yyyy(mm_img, 'mm', _net)

    plt.imshow(yyyy_img, cmap='gray')
    plt.show()
    cv2.imwrite('noise_yyyy.png', yyyy_img)
    y = recognise_dd_mm_yyyy(yyyy_img, 'yyyy', _net)

    return d, m, y

if __name__ == '__main__':
    d, m, y = recognise_dob(argv[1])
    print('DoB: {0:02}-{1:02}-{2:4}'.format(d, m, y))