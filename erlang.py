import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang

#cumulative density function
n = np.zeros((256), dtype=float)  # input image frequency variable
n2 = np.zeros((256), dtype=float)  # erlang distribution frequency variable
prob = np.zeros((256), dtype=float)  # input image probability variable
# erlang distribution probability variable
prob2 = np.zeros((256), dtype=float)
icdf = np.zeros((256), dtype=int)  # input image cdf variable
enlcdf = np.zeros((256), dtype=int)  # erlang distribution cdf variable
#input image
img = cv2.imread('image/input.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('input image', img)
cv2.waitKey(0)

plt.hist(img.ravel(), 256, [0, 255])
plt.show()

z = img.shape[0]*img.shape[1]


erl = erlang.rvs(20, scale=2,  size=(img.shape[0], img.shape[1]))
#print(g)
#round up and type cust to int from float
erl = np.round(erl).astype(int)
# make the range between 0-255
erl[erl > 255] = 255
erl[erl < 0] = 0

plt.hist(erl.ravel(), 256, [0, 255])
plt.show()


x = 0

#find the frequency in input image
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        n[img[i][j]] = n[img[i][j]]+1


#  find the frequency in erlang distribution
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        n2[erl[i][j]] = n2[erl[i][j]]+1

# use erlang equalization in input image
for i in range(0, 256):
    prob[i] = n[i]/(z)
    x = x+prob[i]
    icdf[i] = round(255*x)


x = 0
# use histogram equalization in erlang distribution
for i in range(0, 256):
    prob2[i] = n2[i]/(z)
    x = x+prob2[i]
    enlcdf[i] = round(255*x)


#finally apply histrogram matching
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        val = img[i][j]
        for k in range(256):
            if enlcdf[k] >= icdf[val]:
                res = k
                img.itemset((i, j), res)
                break
#show the output histogram and image
plt.hist(img.ravel(), 256, [0, 255])
plt.show()
cv2.imshow('output image', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
