import math
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import sys
import pylab
from matplotlib.widgets import Slider


# Because this is easier to write and read
def range_inc(start, end): return range(start, end+1)

# This code has been broken up into a bunch of functions for ease of testing

# Function for calculating the laplacian of the gaussian at a given point and with a given variance


def l_o_g(x, y, sigma):
	# Formatted this way for readability
	nom = ((y**2)+(x**2)-2*(sigma**2))
	denom = ((2*math.pi*(sigma**6)))
	expo = math.exp(-((x**2)+(y**2))/(2*(sigma**2)))
	return nom*expo/denom


# Create the laplacian of the gaussian, given a sigma
# Note the recommended size is 7 according to this website http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
# Experimentally, I've found 6 to be much more reliable for images with clear edges and 4 to be better for images with a lot of little edges
def create_log(sigma, size=7):
	w = math.ceil(float(size)*float(sigma))

	# If the dimension is an even number, make it uneven
	if(w % 2 == 0):
		print ("even number detected, incrementing")
		w = w + 1

	# Now make the mask
	l_o_g_mask = []

	w_range = int(math.floor(w/2))
	print ("Going from " + str(-w_range) + " to " + str(w_range))
	for i in range_inc(-w_range, w_range):
		for j in range_inc(-w_range, w_range):
			l_o_g_mask.append(l_o_g(i, j, sigma))
	l_o_g_mask = np.array(l_o_g_mask)
	l_o_g_mask = l_o_g_mask.reshape(w, w)
	return l_o_g_mask

# Convolute the mask with the image. May only work for masks of odd dimensions


def convolve(image, mask):
	width = image.shape[1]
	height = image.shape[0]
	w_range = int(math.floor(mask.shape[0]/2))

	res_image = np.zeros((height, width))

	# Iterate over every pixel that can be covered by the mask
	for i in range(w_range, width-w_range):
		for j in range(w_range, height-w_range):
			# Then convolute with the mask
			for k in range_inc(-w_range, w_range):
				for h in range_inc(-w_range, w_range):
					res_image[j, i] += mask[w_range+h, w_range+k]*image[j+h, i+k]
	return res_image

# Find the zero crossing in the l_o_g image


def z_c_test(l_o_g_image):
	z_c_image = np.zeros(l_o_g_image.shape)

	# Check the sign (negative or positive) of all the pixels around each pixel
	for i in range(1, l_o_g_image.shape[0]-1):
		for j in range(1, l_o_g_image.shape[1]-1):
			neg_count = 0
			pos_count = 0
			for a in range_inc(-1, 1):
				for b in range_inc(-1, 1):
					if(a != 0 and b != 0):
						if(l_o_g_image[i+a, j+b] < 0):
							neg_count += 1
						elif(l_o_g_image[i+a, j+b] > 0):
							pos_count += 1

			# If all the signs around the pixel are the same and they're not all zero, then it's not a zero crossing and not an edge.
			# Otherwise, copy it to the edge map.
			z_c = ((neg_count > 0) and (pos_count > 0))
			if(z_c):
				z_c_image[i, j] = 1

	return z_c_image

# Apply the l_o_g to the image


def run_l_o_g(bin_image, sigma_val, size_val):
	# Create the l_o_g mask
	print("creating mask")
	l_o_g_mask = create_log(sigma_val, size_val)

	# Smooth the image by convolving with the LoG mask
	print ("smoothing")
	l_o_g_image = convolve(bin_image, l_o_g_mask)

	# Display the smoothed imgage
	blurred = fig.add_subplot(1, 4, 2)
	blurred.imshow(l_o_g_image, cmap=cm.gray)

	# Find the zero crossings
	print ("finding zero crossings")
	z_c_image = z_c_test(l_o_g_image)
	print (z_c_image)

	#Display the zero crossings
	edges = fig.add_subplot(1, 4, 3)
	edges.imshow(z_c_image, cmap=cm.gray)
	print ("displaying")
	pylab.show()
	print ('done updating')


# Code that is executed once a slider is updated
def update(val):
	run_l_o_g(bin_image, sigma.val, int(size.val))
	print ("update")


# Create the sliders
axcolor = 'lightgoldenrodyellow'
axsigma = pylab.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axsize = pylab.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sigma = Slider(axsigma, 'Variance', 0.5, 3, valinit=1.4)
size = Slider(axsize, 'Size', 1, 10.0, valinit=5)
sigma_val = sigma.val
size_val = int(size.val)
sigma.on_changed(update)
size.on_changed(update)

# Load the raw file, which is in a very weird format
#f = open('lamp.raw','rb')
#f = open('leaf.raw','rb')
f = open('input.png', 'rb')
#f = open('img335.raw','rb')
#f = open('cana.raw','rb')
# Because the byte order is weird
a = f.read(1)
b = f.read(1)
# First line is rows
rows = int((b+a).encode('hex'), 16)
a = f.read(1)
b = f.read(1)
# Second line is columns
cols = int((b+a).encode('hex'), 16)
# Last byte is encoding, but we're just going to ignore it
f.read(1)
# And everything else is 8 bit encoded, so let's load it into numpy
bin_image = np.fromstring(f.read(), dtype=np.uint8)
# Change the shape of the array to the actual shape of the picture
bin_image.shape = (cols, rows)
# display it with matplotlib
fig = pylab.figure()
original = fig.add_subplot(1, 4, 1)
original.imshow(bin_image, cmap=cm.gray)
# run the laplacian of gaussian edge detector to create the other images
run_l_o_g(bin_image, sigma_val, size_val)

print ("done")
