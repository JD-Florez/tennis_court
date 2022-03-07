This is the readme file for tenniscourt.py, which transforms the two tennis court halves in the video frame into a birdseye view of the tennis court. The file has several functions:
    - **maskcourt**: creates a mask and isolates the in-bounds area of the court from the rest of the image using the blue color
    - **separate**: splits the left and right camera perspectives from each other
    - **morph**: eliminates non-court image objects and closes gaps within the court
    - **maxcontour**: returns a mask containing only the largest contour, the court
    - **filterpts**: two functions, which both filter and order the detected contour points for warping (for left and right sides of the court)
    - **birdseyetransform**: performs the warping needed for the birdseye view

The **main** function reads an image and performs the needed steps to create a birds eye view from it. It assumes that the image is composed of two perspectives that meet evenly in the exact middle of the image. However, this approach works only for a single image and should be expanded to a loop that takes the most recent video frame and performs this warping.
