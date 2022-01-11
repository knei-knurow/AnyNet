import cv2

class Camera:
    def __init__(self, camID, name, interface="csi", mode=3, width=1280, height=720, framerate=30, flipMethod=0, displayFormat="BGR"):
        self.camID = camID,
        self.name = name
        self.interface = interface
        self.mode = mode
        self.width = width
        self.height = height
        self.framerate = framerate
        
        self.flipMethod = flipMethod
        self.displayWidth = width
        self.displayHeight = height
        self.displayFormat = displayFormat

        self.source =  ("nvarguscamerasrc sensor-id=%d sensor-mode=3 !"
        "video/x-raw(memory:NVMM), "
        "width=(int)1920, "
        "height=(int)1080, "
        "format=(string)NV12, "
        "framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink" % self.camID)		# cam ID
            

        self.video = cv2.VideoCapture(self.source)
        while not self.video.isOpened():
            pass


    def shot(self):
        _, self.frame = self.video.read()
        return self.frame
    
