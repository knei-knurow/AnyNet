from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
from cams import Camera
from o3d import GrabO3D300
import cv2 as cv

def main():
    
    #o3d = GrabO3D300()
    left = Camera(0, "left")
    right = Camera(1, "right")

    while True:
        #amp, dist = o3d.get_normalized_images()
        imleft  = left.shot()
        imright = right.shot()

        cv.imshow("left", imleft)
        cv.imshow("right", imright)
        #cv.imshow("amp", amp)
        #cv.imshow("dist", dist)

if __name__ == '__main__':
    main()