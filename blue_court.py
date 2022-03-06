# import the necessary packages
import numpy as np
import cv2
import opencv_wrapper as cvw
import imutils


def findvert(img):
    vertical = np.copy(img)
    rows = vertical.shape[0]
    # We are now looking for the largest vertical edge, which will be along the center of the image
    verticalsize = rows // 10
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    return np.where(vertical != [0])

def separate(img, sr, sc, er, ec, assignleft):
    # Create a black image
    base = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    if assignleft:
        small_img = img[sr:er, sc:ec]
        base[0:small_img.shape[0], 0:small_img.shape[1]] = small_img
    else:
        small_img = img[sr:er, sc:ec]
        base[0:base.shape[0], small_img.shape[1]:base.shape[1]] = small_img
    return base

def getcontours(img):
    contours = []
    cnts = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        #cv2.arcLength(curve, closed)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) > 2 :
            contours.append = approx
    return contours

# construct the argument parse and parse the arguments
# load the image
image = cv2.imread("tennis.jpg")
height, width = image.shape[:2]
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(image,(3,3), sigmaX=0, sigmaY=0)
cv2.imshow("Image", img_blur)
cv2.waitKey(0)

hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

# target blue = 235,182,156
lower_blue = np.array([100,40,40]) #(hue value ~ 110)
upper_blue = np.array([255,255,255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('image', image)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows