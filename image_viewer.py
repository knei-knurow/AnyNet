from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
from cams import Camera
from o3d import GrabO3D300
import cv2 as cv
import numpy as np

def main():
    
    o3d = GrabO3D300()
    left = Camera(0, "left")
    right = Camera(1, "right")

    while True:
        amp, dist = o3d.get_normalized_images()
        amp_visualization = ((1-amp) * 255).astype(np.uint8)
        dist_visualization = ((1-dist) * 255).astype(np.uint8)

        amp_visualization = cv.applyColorMap(amp_visualization, cv.COLORMAP_JET)
        dist_visualization = cv.applyColorMap(dist_visualization, cv.COLORMAP_JET)

        imleft  = left.shot()
        imright = right.shot()

        cv.imshow("left", imleft)
        cv.imshow("right", imright)
        
        cv.imshow("amp", amp_visualization)
        cv.imshow("dist", dist_visualization)
           
        if cv.waitKey(1) == ord('q'):   #  IMPORTANT. CAMS DONT WORK WITHOUT THIS
            pass                        #  DO NOT DELETE

if __name__ == '__main__':
    main()