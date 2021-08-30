import cv2
import numpy as np
import pyrealsense2 as rs
from threading import Thread
from collections import deque

class normalCamThread():

    def __init__(self, resX, resY, cameraID, deque_size):
        self.resX = resX; self.resY = resY
        self.cap = cv2.VideoCapture(cameraID)

        self.deque = deque(maxlen=deque_size)   # Initialize deque used to store frames read from the stream
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True

        # Check if the UV camera opened successfully
        if self.cap.isOpened() == False:
            print("Error opening video stream or file")

        #change webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resX)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resY) 

    def start(self):
        # Start background frame grabbing
        self.get_frame_thread.start() 

    def stop(self):
        self.cap.release()

    def get_frame(self):
        while True:            
            if self.cap.isOpened():
                # Read next frame from stream and insert into deque
                ret, image = self.cap.read()  #uv camera
                if ret:
                    self.deque.append(image)
                
    def get_video_frame(self):
        if len(self.deque) > 0:
            return self.deque[-1]
        else:
            # print ('no frames ;(')
            return None 

# class rsCamThread():

#     def __init__(self, resX, resY, deque_size):
#         # Create a pipeline
#         self.pipeline = rs.pipeline()

#         #Create a config and configure the pipeline to stream
#         #  different resolutions of color and depth streams
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.depth, resX, resY, rs.format.z16, 30)  #GETS BETTER DEPTH READINGS !!
#         self.config.enable_stream(rs.stream.color, resX, resY, rs.format.bgr8, 30)  #GETS BETTER DEPTH READINGS !!

#         self.profile = None
#         self.queue = rs.frame_queue(50)
#         self.colaColor = None 
#         self.colaDepth = None

#         self.get_frame_thread = Thread(target=self.get_frame, args=())         
#         self.get_frame_thread.daemon = True

#     def start(self):
#         # Start background frame grabbing
#         self.get_frame_thread.start() 

#     def stop(self):
#         self.pipeline.stop()

#     def get_frame(self):
#          # Start streaming
#         self.profile = self.pipeline.start(self.config, self.queue)

#         while True:     
                   
#             frames = self.queue.wait_for_frame()
            
#             cFrame = frames.get_color_frame()
#             dFrame = frames.get_depth_frame()

#             if not cFrame or not dFrame:
#                 continue

#             print ('adentro rs')
#             self.colaColor = np.asanyarray(cFrame.get_data())
#             self.colaDepth = np.asanyarray(dFrame.get_data())

#     def get_color_frame(self):
#         return self.colaColor

#     def get_depth_frame(self):
#         return self.colaDepth

#     def get_profile(self):
#         return self.profile
        


