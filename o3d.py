from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
import o3d3xx
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


imageWidth = 352 
imageHeight = 264

class GrabO3D300():
    def __init__(self,data):
        self.data = data
        self.Amplitude = np.zeros((imageHeight,imageWidth))
        self.Distance = np.zeros((imageHeight,imageWidth))

    def read_next_frame(self):
        result = self.data.read_next_frame()
        self.amplitude = np.frombuffer(result['amplitude_image'],dtype='uint16')
        self.amplitude = self.Amplitude.reshape(imageHeight,imageWidth)
        self.distance = np.frombuffer(result['distance_image'],dtype='uint16')
        self.distance = self.Distance.reshape(imageHeight,imageWidth)
        self.illu_temp = 20.0

    def get_amp(self):
        amp_max = float(max(np.max(self.amplitude),1))
        self.amplitude = self.amplitude / amp_max
        return self.amplitude
    
    def get_dist(self):
        dist_max = float(max(np.max(self.distance),1))
        self.distance = self.distance / dist_max
        return self.distance

    def get_images(self):
        self.read_next_frame()
        return self.get_amp(), self.get_dist()

def main():
    address = sys.argv[1]
    camData = o3d3xx.FormatClient(address, 50010,
        o3d3xx.PCICFormat.blobs("amplitude_image", "distance_image"))

    fig = plt.figure()
    grabber = GrabO3D300(camData)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    imAmplitude = ax1.imshow(np.random.rand(imageHeight,imageWidth))
    imDistance = ax2.imshow(np.random.rand(imageHeight,imageWidth))
    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True, fargs = [grabber,imAmplitude,imDistance])
    plt.show()

if __name__ == '__main__':
    main()
