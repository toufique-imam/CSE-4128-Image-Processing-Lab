import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m


def normalize(img):
    # img = np.array(img, dtype='float32')
    nImg = np.zeros(img.shape)  # , dtype='uint8')

    max_ = img.max()
    min_ = img.min()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            nImg[i][j] = (img[i][j]-min_)/(max_-min_) * 255

    print(img.min(), img.max())
    print(nImg.min(), nImg.max())
    print()

    return np.array(nImg, dtype='uint8')
    # return nImg


img = cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("1st.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("einstein.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("input", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

guass = 5
sig = 1

gRow = np.array([[m.exp(-(i*i)/(2*sig*sig)) / 2*m.pi*sig *
                  sig for i in range(-(guass//2), (guass//2)+1)]])
gMat = np.matmul(gRow.T, gRow)
# gMat = gMat / np.sum(gMat)

# gMat = np.array([[1, 2, 1],
#                  [2, 4, 2],
#                  [1, 2, 1]])

# for schhol project
img_ = np.array(img, dtype='float32')

gImg_ = cv2.filter2D(img_, -1, gMat)
gImg = normalize(gImg_)
cv2.imshow("Gaussian", gImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

sx = cv2.filter2D(gImg_, -1, sobel_x)
sx_ = normalize(sx)
cv2.imshow("sobel_x", sx_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sy = cv2.filter2D(gImg_, -1, sobel_y)
sy_ = normalize(sy)
cv2.imshow("sobel_y", sy_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# sx = normalize(sx)
# sy = normalize(sy)

# sx = np.array(sx, dtype='float32')
# sy = np.array(sy, dtype='float32')

sobel_ = np.sqrt(sx*sx + sy*sy)
sobel = normalize(sobel_)
# mag = sobel.copy()

ang = np.arctan2(sy, sx)
# sobel = np.array(sobel, dtype='uint8')

cv2.imshow("sobel", sobel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# tr = 60

# sobel[sobel>tr] = 255
# sobel[sobel<=tr] = 0

# cv2.imshow("sobel_threshed", sobel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# non_max_suppression ...
mag = sobel_.copy()

for i in range(mag.shape[0]):
    for j in range(mag.shape[1]):
        if (ang[i][j] >= (-m.pi/8) and ang[i][j] <= (m.pi/8)) or (ang[i][j] >= (7*m.pi/8) and ang[i][j] <= (-7*m.pi/8)):
            # left-right
            if j == 0:
                pass
            elif j == mag.shape[1]-1:
                pass
            else:
                if (mag[i][j] < mag[i][j-1]) or (mag[i][j] < mag[i][j+1]):
                    mag[i][j] = 0

        elif (ang[i][j] >= (-3*m.pi/8) and ang[i][j] <= (-m.pi/8)) or (ang[i][j] >= (5*m.pi/8) and ang[i][j] <= (7*m.pi/8)):
            # top_left-bottom_right
            if i == 0 or i == mag.shape[0]-1 or j == 0 or j == mag.shape[1]-1:
                pass
            else:
                if (mag[i][j] < mag[i+1][j-1]) or (mag[i][j] < mag[i-1][j+1]):
                    mag[i][j] = 0

        elif (ang[i][j] >= (-7*m.pi/8) and ang[i][j] <= (-5*m.pi/8)) or (ang[i][j] >= (m.pi/8) and ang[i][j] <= (3*m.pi/8)):
            # top_right-bottom_left
            if i == 0 or i == mag.shape[0]-1 or j == 0 or j == mag.shape[1]-1:
                pass
            else:
                if (mag[i][j] < mag[i-1][j-1]) or (mag[i][j] < mag[i+1][j+1]):
                    mag[i][j] = 0

        elif (ang[i][j] >= (-5*m.pi/8) and ang[i][j] <= (-3*m.pi/8)) or (ang[i][j] >= (3*m.pi/8) and ang[i][j] <= (5*m.pi/8)):
            # top-bottom
            if i == 0:
                pass
            elif i == mag.shape[0]-1:
                pass
            else:
                if (mag[i][j] < mag[i-1][j]) or (mag[i][j] < mag[i+1][j]):
                    mag[i][j] = 0

mag_ = normalize(mag)

cv2.imshow("non max suppressed", mag_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

high_tr = 70
low_tr = 40

high_pix = 255
low_pix = 100

threshed = mag_.copy()

threshed[threshed >= high_tr] = high_pix
threshed[np.logical_and(threshed < high_tr, threshed >= low_tr)] = low_pix
threshed[threshed < low_tr] = 0

cv2.imshow("high-low threshholded", threshed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Hysteresis
hyst = threshed.copy()

for i in range(1, hyst.shape[0]-1):
    for j in range(1, hyst.shape[1]-1):
        if hyst[i][j] == low_pix:
            if np.any(hyst[i-1:i+2][j-1:j+2] == high_pix):
                hyst[i][j] = high_pix
            else:
                hyst[i][j] = 0

cv2.imshow("Canny Edge", hyst)
cv2.waitKey(0)
cv2.destroyAllWindows()
