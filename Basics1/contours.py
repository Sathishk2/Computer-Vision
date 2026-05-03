#pylint:disable=no-member
# Contours are important for shape analysis, object detection, and recognition

import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# grab the edges of the image
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# binarize an image, meaning if the density of the pixel is below 125, it'll be set to 0. if it is above 125, it is set to white. 
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# find contours of the image: 
# hierarchies = inside rectangle, theres a square. inside square, theres circle. 
# RETR_TREE = hierarchical contours, RETR_external = external contours - only the ones on the outside, RETR_LIST = all the contours in the image.
# CHAIN_APPROX_SIMPLE = extract the contours that makes the most sense. if you have a line, it'll only get 2 end points of the line. 
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

# takes a blank image to draw over, takes a list of contours, index of how many contours i want in an image, color, and thickness
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)