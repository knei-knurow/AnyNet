import numpy as np
import cv2
import sys
import argparse
import yaml
import logging

parser = argparse.ArgumentParser(description='Test undistortion')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
parser.add_argument('path', type=str, help="Path to config file")

args = parser.parse_args()
path = args.path

with open(path, 'r') as file:
    config = yaml.load(file)

cv2.namedWindow('Real', cv2.WINDOW_NORMAL)
cv2.namedWindow('Undistorted', cv2.WINDOW_NORMAL)

gststring = ("nvarguscamerasrc sensor-id={id}"
            " ! video/x-raw(memory:NVMM), format=NV12, width={width}, height={height} "
            " ! nvvidconv ! video/x-raw, width={width}, height={height}, "
            " format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR "
            " ! videoconvert ! appsink ").format(id=config["id"], width=config["width"], height=config["height"])

cap = cv2.VideoCapture(gststring)

camera_matrix = config["camera_matrix"]
distortion_coeffs = config["distortion_coeffs"]
new_camera_matrix = config["optimal_camera_matrix"]

print(type(camera_matrix))
print(type(distortion_coeffs))
print(type(new_camera_matrix))


while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        continue
    frame_undistorted = cv2.undistort(frame,
    camera_matrix, 
    distortion_coeffs,
    newCameraMatrix = new_camera_matrix 
    )    
    
    key = cv2.waitKey(2) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

    cv2.imshow('Real', frame)
    cv2.imshow('Undistorted', frame_undistorted)