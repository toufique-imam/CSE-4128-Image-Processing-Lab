from PIL import Image
from scipy import signal
import numpy as np
import cv2
from matplotlib import pyplot as plt
import image_lib
import convulation_as_multiplication as conmul


def showimg(img_out):
    plt.imshow(img_out, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def cv_showimg(img_out, label="image"):
    cv2.imshow(label, img_out)
    cv2.waitKey(0)


OLD_PATH = "playtv_notification.png"
NEW_PATH = "playtv_notification_edit.png"

image = cv2.imread(OLD_PATH, -1)

#image[np.where((image == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

im = Image.open(OLD_PATH)
pixels = im.load()

width, height = im.size
for x in range(width):
    for y in range(height):
        r, g, b, a = pixels[x, y]
        if (r, g, b) == (255, 255, 255):
            pixels[x, y] = (0, 0, 0 ,a)
im.save(NEW_PATH)
        
    

