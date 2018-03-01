import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from recognise_id_number import validate, get_digit_contours, square_padding, crop_id

def crop(img, filename):
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
    if (is_valid == True):
        # Specify the bounding rectangle for each of selected contour
        horizontal_top_lefts = [0] * num_digits
        rects = [None] * num_digits
        for i in range(num_digits):
            x, y, w, h = cv2.boundingRect(digit_contours[i])
            rects[i] = (x, y, w, h)
            horizontal_top_lefts[i] = x
        save_dir = 'cropped'
        for i in range(num_digits):
            x = rects[i][0]
            y = rects[i][1]
            w = rects[i][2]
            h = rects[i][3]
            img_name = os.path.join(save_dir, filename+'_'+str(i+1)+'.png')
            cv2.imwrite(img_name, square_padding(img[y:y + h + 1, x:x + w + 1]))
            print(img_name + " cropped!")

if __name__ == "__main__":
    # file_path = 'number_id_test/many_digit_groups'
    # filename = '841289701567_0901315908_023841227_1_2016032415243805.jpgcardnumber'
    
    img_dir = '../number_all/id_number_combined'
    filenames = os.listdir(img_dir)
    for filename in filenames:
        print('Processing ' + filename)
        id_only_img = crop_id(os.path.join(img_dir, filename[:-4] + '.png'))
        crop(id_only_img, filename)