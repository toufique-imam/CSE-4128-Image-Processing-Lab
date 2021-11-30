import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import os


def cv_showimg(img_out, label="image"):
    cv2.imshow(label, img_out)
    cv2.waitKey(0)


def crop_top(img):
    height = img.shape[0]
    width = img.shape[1]
    idx = 0
    while idx < height:
        f1 = 1
        idy_now = 0
        while idy_now < width:
            if(img[idx][idy_now][0] != 255 and img[idx][idy_now][1] != 255 and img[idx][idy_now][2] != 255):
                f1 = 0
                break
            idy_now += 1
        if(f1==0):
            break
        idx += 1
    return img[idx:]


def crop_down(img):
    height = img.shape[0]
    width = img.shape[1]
    idx = height - 1
    while idx > -1:
        f1 = 1
        idy_now = 0
        while idy_now < width:
            if(img[idx][idy_now][0] != 255 and img[idx][idy_now][1] != 255 and img[idx][idy_now][2] != 255):
                f1 = 0
                break
            idy_now += 1
        if(f1==0):
            break
        idx -= 1
    return img[0:idx]


def crop_left(img):
    height = img.shape[0]
    width = img.shape[1]
    idy = 0
    while idy < width:
        f1 = 1
        idx_now = 0
        while idx_now < height:
            if(img[idx_now][idy][0] != 255 and img[idx_now][idy][1] != 255 and img[idx_now][idy][2] != 255):
                f1 = 0
                break
            idx_now += 1
        if(f1==0):
            break
        idy += 1

    idy = max(0, idy-1)
    img_new = []
    for i in range(0 , height):
        img_temp = []
        for j in range(idy , width):
            img_temp.append(img[i][j])
        img_new.append(img_temp)
    return np.array(img_new)
    # return img[:, :, :, idy]


def crop_right(img):
    height = img.shape[0]
    width = img.shape[1]
    # print(img.shape , "\n" , img)
    idy = width - 1
    while idy > -1:
        f1 = 1
        idx_now = 0
        while idx_now < height:
            if(img[idx_now][idy][0] != 255 and img[idx_now][idy][1] != 255 and img[idx_now][idy][2] != 255):
                f1 = 0
                break
            idx_now += 1
        if(f1==0):
            break
        idy -= 1

    idy = min(width , idy+1)
    img_new = []
    for i in range(0, height):
        img_temp = []
        for j in range(0, idy):
            img_temp.append(img[i][j])
        img_new.append(img_temp)
    return np.array(img_new)


def corner_crop(img):
    height = img.shape[0]
    width = img.shape[1]
    idx = 0
    while idx < height/2:
        f1 = 1
        f2 = 1
        idx_now = idx
        idy_now = idx
        w_now = width-idx
        h_now = height-idx
        while(idy_now < w_now):
            if(img[idx_now][idy_now][0] != 255 and img[idx_now][idy_now][1] != 255 and img[idx_now][idy_now][2] != 255):
                f1 = 0
                break
            rev_idx_x = height - idx_now - 1
            if(img[rev_idx_x][idy_now][0] != 255 and img[rev_idx_x][idy_now][1] != 255 and img[rev_idx_x][idy_now][2] != 255):
                f1 = 0
                break
            idy_now += 1
        if(f1 == 0):
            break

        idx_now = idx
        idy_now = idx
        while(idx_now < h_now):
            if(img[idx_now][idy_now][0] != 255 and img[idx_now][idy_now][1] != 255 and img[idx_now][idy_now][2] != 255):
                f2 = 0
                break
            rev_y = height - idy_now - 1
            if(img[idx_now][rev_y][0] != 255 and img[idx_now][rev_y][1] != 255 and img[idx_now][rev_y][2] != 255):
                f2 = 0
                break
            idx_now += 1
        if(f2 == 0):
            break

        idx += 1
    img_new = []

    i = idx
    while(i < height-idx):
        image_temp = []
        j = idx
        while(j < width-idx):
            image_temp.append(img[i][j])
            j += 1
        img_new.append(image_temp)
        i += 1
    img_new = np.array(img_new)

    return img_new

    cv2.imwrite(pathSave, img_new)
    print(idx, img.shape, img_new.shape, pathSave)


def white_trimmer(path , parent_path):
    pathLoad = parent_path+"/"+path
    if not os.path.exists(parent_path+"_new"):
        os.makedirs(parent_path+"_new")
    pathSave = parent_path+"_new/"+path
    img = cv2.imread(pathLoad)
    shape_prev = img.shape
    img = corner_crop(img)
    shape_prev_c = img.shape
    img = crop_top(img)
    shape_prev_t = img.shape
    img = crop_down(img)
    shape_prev_d = img.shape
    img = crop_left(img)
    shape_prev_l = img.shape
    img = crop_right(img)
    cv2.imwrite(pathSave, img)
    print(shape_prev ,shape_prev_c , shape_prev_t , shape_prev_d , shape_prev_l, img.shape, pathSave)


mypath = input("input folder name : ")
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(len(onlyfiles))
_ii = 0
for i in onlyfiles:
    print(_ii)   
    white_trimmer(i , mypath)
    _ii+=1
