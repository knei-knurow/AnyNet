from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
import o3d3xx
import numpy as np


imageWidth = 352 
imageHeight = 264

class GrabO3D300():
    def __init__(self, addr="192.168.1.240", port=50010):
        self.format =  o3d3xx.PCICFormat.blobs("amplitude_image", "distance_image")
        self.data = o3d3xx.FormatClient(addr, port, self.format)
        self.amplitude = np.zeros((imageHeight,imageWidth))
        self.distance = np.zeros((imageHeight,imageWidth))

    def read_next_frame(self):
        result = self.data.readNextFrame()
        self.amplitude = np.frombuffer(result['amplitude_image'],dtype='uint16').astype(np.double)
        self.amplitude = self.amplitude.reshape(imageHeight,imageWidth)
        self.distance = np.frombuffer(result['distance_image'],dtype='uint16').astype(np.double)
        self.distance = self.distance.reshape(imageHeight,imageWidth)
        self.illu_temp = 20.0

    def get_amp(self):
        amp_max = float(np.amax(self.amplitude))
        self.amplitude = self.amplitude / amp_max
        return self.amplitude
    
    def get_dist(self):
        dist_max = float(np.amax(self.distance))
        self.distance = self.distance / np.double(dist_max)
        return self.distance

    def get_normalized_images(self):
        self.read_next_frame()
        return self.get_amp(), self.get_dist()

