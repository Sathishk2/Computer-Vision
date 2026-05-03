#pylint:disable=no-member

import cv2 as cv

img = cv.imread('../Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

# Gaussian Blur (3,3) = meaning kernal or window. kernal size is the number of rows and columns. 
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)

# Median Blur
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# Bilateral (retains the edges in the image) (img, diameter, more sigma (more colors in the neighborhood are considered), further pixels from the central pixels will influence the blurring calculation, )
bilateral = cv.bilateralFilter(img, 10, 35, 25 )
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)