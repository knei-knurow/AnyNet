import yaml

with open("config.yml", "r") as config_file:
    cfg = yaml.safe_load(config_file)

host = cfg["host"]
port = cfg["port"]
debug = cfg["debug"]
sources = cfg["sources"]
path = cfg["path"]
is_remote = cfg["remote"]


with open("rovercamera/config/stereo.yml", "r") as rovercamera_config_file:
    rovercamera_cfg = yaml.full_load(rovercamera_config_file)

from rovercamera import RoverCamera

import cv2 as cv
import logging
import uuid
import os


# urllib debug messages
if debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


def get_rovercamera(name):
    # logging.info("Connecting to: ", name)
    return rovercamera.RoverCamera(name, remote=is_remote, host=host, port=port, config=rovercamera_cfg)




logging.info("Host: ", host, " port: ", port)

cameras = []
for source in sources:
    cameras.append(get_rovercamera(source))

dataset_path = path + str(uuid.uuid4())

os.mkdir(dataset_path)

for i in range(len(cameras)):
    print(dataset_path + "/" + str(i))
    os.mkdir(dataset_path + "/" + str(i))

logging.info("Created dataset: " + dataset_path)

count = 0

while True:
    count += 1

    # Fetch images
    logging.info("Fetching images")

    images = []
    for camera in cameras:
        images.append(camera.get_frame())

    preview_images = []
    for image in images:
        preview_images.append(cv.resize(image, (360, 240)))

    preview = cv.vconcat(preview_images)

    # Show
    cv.imshow("Preview", preview)

    if cv.waitKey(1) == ord('s'):
        for j in range(len(cameras)):
            cv.imwrite(dataset_path + "/" + str(j) +
                       "/" + str(count) + ".jpg", images[j])
            logging.info("Saved! ")




