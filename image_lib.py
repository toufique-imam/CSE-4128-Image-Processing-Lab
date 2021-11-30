from scipy.linalg import toeplitz
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def showImageList(imgList):
    fig = plt.figure(figsize=(20,20))
    columns = len(imgList)
    rows = 1
    for i in range(len(imgList)):
        fig.add_subplot(rows,columns,i+1)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.imshow(imgList[i], cmap='gray', interpolation='bicubic')

    plt.show()

def convolve_np_zero(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = int((F_height - 1) / 2)
    W = int((F_width - 1) / 2)
    out = np.zeros((X_height, X_width))
    #out = X.copy()
    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            idxx = F_height -1
            for k in np.arange(-H, H+1):
                idxy = F_width - 1
                for l in np.arange(-W, W+1):
                    a = X[i+k, j+l]
                    w = F[idxx, idxy]
                    idxy=idxy-1
                    sum += (w * a)
                idxx=idxx-1
            out[i, j] = sum
    return out

def convolve_np_copy(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = int((F_height - 1) / 2)
    W = int((F_width - 1) / 2)
    #out = np.zeros((X_height, X_width))
    out = X.copy()
    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    a = X[i+k, j+l]
                    w = F[H+k, W+l]
                    sum += (w * a)
            out[i, j] = sum
    return out

def max_filter(X, F_height, F_width):
    X_height = X.shape[0]
    X_width = X.shape[1]

    H = int((F_height - 1) / 2)
    W = int((F_width - 1) / 2)
    out = np.zeros((X_height, X_width))

    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = -255
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    a = X[i+k, j+l]
                    sum = max(sum, a)
            out[i, j] = sum
    return out

def min_filter(X, F_height, F_width):
    X_height = X.shape[0]
    X_width = X.shape[1]

    H = int((F_height - 1) / 2)
    W = int((F_width - 1) / 2)
    out = np.zeros((X_height, X_width))

    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 255
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    a = X[i+k, j+l]
                    sum = min(sum, a)
            out[i, j] = sum
    return out

def median_filter(X, F_height, F_width):
    X_height = X.shape[0]
    X_width = X.shape[1]

    H = int((F_height - 1) / 2)
    W = int((F_width - 1) / 2)
    out = np.zeros((X_height, X_width))

    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = []
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    a = X[i+k, j+l]
                    sum.append(a)
            sum.sort()
            if((len(sum) % 2) == 1):
                out[i, j] = sum[len(sum)//2+1]
            else:
                out[i, j] = sum[len(sum)//2]
    return out

def __calculate_laplace(x,y,sigma):
    a=(x*x+y*y-2.0*sigma*sigma)
    denom = (2*math.pi*(sigma**6))
    b = (x*x+y*y)/(2.0*sigma*sigma)
    c = math.exp(-b)
    return a*c/denom

def LoG(n,sigma):
    w = math.ceil(n*sigma)
    if(w%2==0):
        w=w+1
    matrix = []
    r = int(math.floor(w/2))
    for i in range(-r,r+1):
        for j in range(-r,r+1):
            matrix.append(__calculate_laplace(i,j,sigma))
    matrix = np.array(matrix)
    matrix = matrix.reshape(w,w)
    return matrix

def __calculate_gauss(x,y,sigma):
    a=1.0/(math.sqrt(2*math.pi*sigma*sigma))
    b = (x*x+y*y)/(2*sigma*sigma)
    c = math.exp(-b)
    return a*c;

def gauss_sigma(n,sigma):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = __calculate_gauss(i, j, sigma)
    return matrix
def tempLoG():
    return (1.0/16)*np.array(
        [[0, 0, -1, 0, 0],
         [0, -1, -2, -1, 0],
         [-1, -2, 16, -2, -1],
         [0, -1, -2, -1, 0],
         [0, 0, -1, 0, 0]])

def prewitt(n):
    Hx = np.zeros((n, n))
    Hy = np.zeros((n, n))

    for i in range(0, n):
        Hx[0][i] = -1
        Hx[n-1][i] = 1
        Hy[i][0] = -1
        Hy[i][n-1] = 1
    return (Hx, Hy)

def standard_average(n):
    out = np.ones((n, n))
    return out

def gauss_weighted_average(n):
    out_x = []
    out_y = []
    val = 1
    for i in range(n//2):
        out_x.append(val)
        out_y.append([val])
        val = val * 2
    out_x.append(val)
    out_y.append([val])
    for j in range(n//2):
        val = val/2
        out_x.append(val)
        out_y.append([val])
    out_x = [out_x]
    out = np.matmul(out_y, out_x)
    return out

def map_img(img, range):
    img = (img / np.max(img)) * range
    return img

def sharp_matrix_5():
    return np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]);
def sharp_matrix_9():
    return np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]);
def sobel_row():
    return np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
def sobel_col():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
def scharr_row():
    return np.array([[-3, 0, +3],
                      [-10, 0, 10],
                      [-3, 0, 3]])
def scharr_col():
    return np.array([[-3, -10, -3],
                     [-10, 0, 10],
                     [3, 10, 3]])
def shiftMatrixRight(n):
    out = np.zeros((n,n))
    out[n//2][n-1]=1
    return out
def shiftMatrixLeft(n):
    out = np.zeros((n,n))
    out[n//2][0]=1
    return out
