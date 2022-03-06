# import the necessary packages
import numpy as np
import cv2
import opencv_wrapper as cvw
import imutils

# this function isolates the court using the blue color of the in-bounds play-area
def maskcourt(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # target blue = 235,182,156
    lower_blue = np.array([100,40,40]) #(hue value ~ 110)
    upper_blue = np.array([255,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    return mask, res

# this function splits left from right
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

# this function closes gaps and eliminates extranous image artifacts
def morph(img):
    #imgblurred = cv2.GaussianBlur(img, (21,21), sigmaX=0, sigmaY=0)
    imgblurred = cv2.medianBlur(img, 17)
    # cv2.imshow("blurred court", imgblurred)
    # cv2.waitKey(0)
    opened = cv2.morphologyEx(imgblurred, cv2.MORPH_OPEN, (31,31))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, (501,501))

    return closed

# this function returns a mask of just the court
def maxcontour(img):
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    area_old = 0
    for cnt in contours:        
        area = cv2.contourArea(cnt)         
        if area > area_old:
            area_old = area
            contour = cnt
    epsilon = 0.01*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    return approx

# this function filters and orders the points, from the left side of the court, for warping
def filterptsleft(pts):
    # Find the bottom left point
    for i in range(pts.shape[0]):
        if i < 1:
            TL = pts[i][:][:]
            TR = pts[i][:][:]
            BL = pts[i][:][:]
            BR = pts[i][:][:]
        else:
            if pts[i][0][0] < BL[0][0]:
                BL = pts[i][:][:]
                #print("New BL:", BL)
    #print("BL", BL)
    count_TL = 0
    # Find the top left point
    for i in range(pts.shape[0]):
        if pts[i][0][0] == BL[0][0]:
            continue
        if pts[i][0][1] <= TL[0][1]:
            if count_TL > 0:
                if pts[i][0][0] < TL[0][0]:
                    TL = pts[i][:][:]
            else:
                TL = pts[i][:][:]
            #print("New TL:", TL)
            count_TL += 1
    #print("TL", TL)
    count_TR = 0
    # Find the top right point
    for i in range(pts.shape[0]):
        if pts[i][0][0] == BL[0][0]:
            continue
        if pts[i][0][0] == TL[0][0] and pts[i][0][1] == TL[0][1]:
            continue
        if pts[i][0][1] <= TR[0][1]:
            if count_TR > 0:
                if pts[i][0][0] > TR[0][0]:
                    TR = pts[i][:][:]
            else:
                TR = pts[i][:][:]
            #print("New TR:", TR)
            count_TR += 1
    #print("TR", TR)
    count_BR = 0
    # Find the bottom right point
    for i in range(pts.shape[0]):
        if pts[i][0][0] == BL[0][0]:
            continue
        if pts[i][0][1] == TL[0][1]:
            continue
        if pts[i][0][1] == TR[0][1]:
            continue
        if pts[i][0][1] >= BR[0][1]:
            if count_BR > 0:
                if pts[i][0][0] < BR[0][0]:
                    BR = pts[i][:][:]
            else:
                BR = pts[i][:][:]
            #print("New BR:", BR)
            count_BR += 1
    #print("BR", BR)
    contour = [TL, TR, BL, BR]
    points = [TL[0], TR[0], BR[0], BL[0]]
    return contour, points

# this function filters and orders the points, from the left side of the court, for warping
def filterptsright(pts):
    
    # Find the bottom right point
    for i in range(pts.shape[0]):
        if i < 1:
            TL = pts[i][:][:]
            TR = pts[i][:][:]
            BL = pts[i][:][:]
            BR = pts[i][:][:]
        else:
            if pts[i][0][0] > BR[0][0]:
                BR = pts[i][:][:]
                #print("New BR:", BR)
    #print("BR", BR)
    count_TR = 0
    # Find the top right point
    for i in range(pts.shape[0]):
        if pts[i][0][0] == BR[0][0]:
            continue
        if pts[i][0][1] <= TR[0][1]:
            if count_TR > 0:
                if pts[i][0][0] > TR[0][0]:
                    TR = pts[i][:][:]
            else:
                TR = pts[i][:][:]
            #print("New TR:", TR)
            count_TR += 1
    #print("TR", TR)
    count_TL = 0
    # Find the top right point
    for i in range(pts.shape[0]):
        if pts[i][0][0] == BR[0][0]:
            continue
        if pts[i][0][0] == TR[0][0] and pts[i][0][1] == TR[0][1]:
            continue
        if pts[i][0][1] <= TL[0][1]:
            if count_TR > 0:
                if pts[i][0][0] < TL[0][0]:
                    TL = pts[i][:][:]
            else:
                TL = pts[i][:][:]
            #print("New TL:", TL)
            count_TL += 1
    #print("TL", TL)
    count_BL = 0
    # Find the bottom left point
    for i in range(pts.shape[0]):
        if pts[i][0][0] == BR[0][0]:
            continue
        if pts[i][0][1] == TR[0][1]:
            continue
        if pts[i][0][1] == TL[0][1]:
            continue
        if pts[i][0][1] >= BL[0][1]:
            if count_BL > 0:
                if pts[i][0][0] > BL[0][0]:
                    BL = pts[i][:][:]
            else:
                BL = pts[i][:][:]
            #print("New BL:", BL)
            count_BL += 1
    #print("BL", BL)
    contour = [TL, TR, BL, BR]
    points = [TL[0], TR[0], BR[0], BL[0]]
    return contour, points

# warps the image using the points returned from filterpts
def birdeyetransform(image, pts):
	rect = pts
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(np.float32(rect), dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

def main():
    # construct the argument parse and parse the arguments
    # load the image
    image = cv2.imread("tennis.jpg")
    cv2.imshow("base image", image)
    cv2.waitKey(0)
    height, width = image.shape[:2]
    #imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(image,(3,3), sigmaX=0, sigmaY=0)

    # find the mask and resulting image for the court when filtering for the color blue
    courtmask, _ = maskcourt(img_blur)
    # cv2.imshow("courtmask", courtmask)
    # Let's get the starting pixel coordiantes (top left of cropped top)
    L_start_row, L_start_col = int(0), int(0)
    R_start_row, R_start_col = int(0), int(width/2)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    L_end_row, L_end_col = int(height), int(width/2)
    R_end_row, R_end_col = int(height), int(width)
    L = separate(courtmask, L_start_row, L_start_col, L_end_row, L_end_col, True)
    R = separate(courtmask, R_start_row, R_start_col, R_end_row, R_end_col, False)

    L_morphed = morph(L)
    L_imgmasked = cv2.bitwise_and(img_blur, img_blur, mask=L_morphed)
    R_morphed = morph(R)
    R_imgmasked = cv2.bitwise_and(img_blur, img_blur, mask=R_morphed)

    L_contour = maxcontour(L_morphed)
    contL, L_pts = filterptsleft(L_contour)
    # img_corners = cv2.drawContours(image, contL, -1, (0,255,0), 3)

    R_contour = maxcontour(R_morphed)
    contR, R_pts = filterptsright(R_contour)
    # img_corners = cv2.drawContours(img_corners, contR, -1, (0,255,0), 3)
    # cv2.imshow("image corners", img_corners)
    # cv2.waitKey(0)

    lwarp = birdeyetransform(image, L_pts)
    rwarp = birdeyetransform(image, R_pts)

    if lwarp.shape[0] >= rwarp.shape[0]:
        final_base = np.zeros((lwarp.shape[0],lwarp.shape[1]+rwarp.shape[1], 3), np.uint8)
    else:
        final_base = np.zeros((rwarp.shape[0],lwarp.shape[1]+rwarp.shape[1], 3), np.uint8)

    print(final_base.shape, lwarp.shape, rwarp.shape,)
    final_base[0:lwarp.shape[0], 0:lwarp.shape[1]] = lwarp
    final_base[0:rwarp.shape[0], lwarp.shape[1]:final_base.shape[1]] = rwarp
    cv2.imshow("warp", final_base)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

