# -*- coding: utf-8 -*-
import os
import cv2

def label_mask(img_file, mask_file):
    img = cv2.imread(img_file)
    mask = cv2.imread(mask_file)
    img = cv2.addWeighted(img,0.6,mask,0.4,0)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask, 0, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #绘制轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    return img


if __name__ == "__main__":
    img_root = "test_data"
    mask_root = "test_result"
    test_label_mask = "test_label_mask"

    for img_file in os.listdir(img_root):
        res = label_mask(
            os.path.join(img_root,  img_file),
            os.path.join(mask_root, img_file)
        )
        cv2.imwrite(os.path.join(test_label_mask, img_file),res)




