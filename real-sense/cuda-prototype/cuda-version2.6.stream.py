#########################################################################
#    CUDA version for speed reasons only the UV camera is used          # 
#    the system works with the RS and both chinese cameras connected    #
#     (introduces RGBd rectification by opencv function )               # 
#########################################################################\

#TODO list
# fix the double images in the rectification   TODO
# fix or cut the rectification warp of the final images.   DONE
# optimize gpumat declarations and uploads           DONE 
# check if the camera taking photos creates a bottle neck  DONE
# remove salt and peper from the rectified images TODO


import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
from cameraThread import normalCamThread
from matplotlib import pyplot as plt  #matplotlib for DEBUG only
from cuda_stream26 import cuStream
from cuda_stream26 import Matrices


#PROGRAM CONSTANTS 
mapx1_uv=None; mapx2_uv=None; mapy1_uv=None; mapy2_uv=None
uvVroi = []
resX = 1280
resY = 720
err = 0.0 #DEBUG
alphaValue = 1.0
fpsCount = 0

def loadCamFiles(fExt, fInt):
    #Load extrinsic matrix variables 
    fext = cv2.FileStorage(fExt, cv2.FILE_STORAGE_READ) 
    R = fext.getNode("R").mat()
    T = fext.getNode("T").mat()
    R1 = fext.getNode("R1").mat()
    R2 = fext.getNode("R2").mat()
    P1 = fext.getNode("P1").mat()
    P2 = fext.getNode("P2").mat()
    Q = fext.getNode("Q").mat()

    #Load intrinsic matrix variables  (only for UV chinese camera)
    fint = cv2.FileStorage(fInt, cv2.FILE_STORAGE_READ) 
    M1 = fint.getNode("M1").mat()  #cameraMatrix[0]  #RGB
    D1 = fint.getNode("D1").mat()  #distCoeffs[0]    #RGB
    M2 = fint.getNode("M2").mat()  #cameraMatrix[1]  #UV
    D2 = fint.getNode("D2").mat()  #distCoeffs[2]    #UV

    r1 = fext.getNode("validRoi1")
    r2 = fext.getNode("validRoi2")
    vroi1 = r1.mat()
    vroi2 = r2.mat()

    #get the SMALLEST rectangle cut 
    uvVroi.append( max(vroi1[0][0], vroi2[0][0]) )  #x
    uvVroi.append( max(vroi1[1][0], vroi2[1][0]) )  #y
    uvVroi.append( min(vroi1[2][0], vroi2[2][0]) )  #w
    uvVroi.append( min(vroi1[3][0], vroi2[3][0]) )  #h
 
    fint.release()
    fext.release()
    return [M1, M2, D1, D2, R1, R2, P1, P2, T]

#calculates the rectify matrices from the camera parameters
def undistortUV(M1, M2, D1, D2, R1, R2, P1, P2):
    # PROBAR ESTA OPCION !!! OPCIONAL 1!!
    # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    #Mat rmap[2][2];
    global mapx1_uv; global mapx2_uv; global mapy1_uv; global mapy2_uv
    mapx1_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy1_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapx2_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy2_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')

    mapx1_uv, mapy1_uv = cv2.initUndistortRectifyMap(M1, D1, R1, P1,(resX, resY), cv2.CV_32F)  #original cv2.CV_16SC2 
    mapx2_uv, mapy2_uv = cv2.initUndistortRectifyMap(M2, D2, R2, P2,(resX, resY), cv2.CV_32F)


#interactive matlab plot
# plt.ion() 

#Objects to get the camera frames in different thread 
iCam = normalCamThread(resX, resY, 4, 90)   #chinese with leds     (infra)
uCam = normalCamThread(resX, resY, 1, 90)  #chinese with no leds  (uvcam)
iCam.start()
uCam.start()

# Create RS pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, resX, resY, rs.format.z16, 30)  #GETS BETTER DEPTH READINGS !!
config.enable_stream(rs.stream.color, resX, resY, rs.format.bgr8, 30)  #GETS BETTER DEPTH READINGS !!
rs_queue = rs.frame_queue(3)  #saves the frames 

# Start streaming/
profile = pipeline.start(config, rs_queue)

# Getting the depth sensor's depth scale (see rs-align example for explanation) 
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, True)
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.0 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
# print("Clipping Distance is: " , clipping_distance)

# Create AN ALGINED OBJECT
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
# align_to = rs.stream.color
# align = rs.align(align_to)

# baseline, F, matrices[M1, M2, D1, D2, R1, R2, P1, P2]
Mats = loadCamFiles('./cam_params/rgb_uv_extrinsics.yml', './cam_params/rgb_uv_intrinsics.yml')  
undistortUV(Mats[0], Mats[1], Mats[2], Mats[3], Mats[4], Mats[5], Mats[6], Mats[7])  # calculate the distortion matrices
base_rgb_uv = abs(float(Mats[8][0])) * abs(float(Mats[0][0][0]))                     # T[1] M1[fx] cause rectified on Y (horizontal cameras)

# CALCULATE THE MATRIX TO CUT THE FINAL IMAGE
# print ("VRoi dimentions: " + str(uvVroi))
pts1 = np.float32([[uvVroi[0],uvVroi[1]], [uvVroi[0]+uvVroi[2], uvVroi[1] ], [uvVroi[0], uvVroi[1]+uvVroi[3]], [uvVroi[0]+uvVroi[2], uvVroi[1]+uvVroi[3]] ])
pts2 = np.float32([[0,0],[resX,0],[0,resY],[resX,resY]])
cutMatrixUv = cv2.getPerspectiveTransform(pts1,pts2)

################# Variables for the RGBD OPENCV registration ###############################
i_profile = profile.get_stream(rs.stream.depth)
intr = i_profile.as_video_stream_profile().get_intrinsics()

depth_cam_matrix = np.zeros((3,3), dtype=np.float)
depth_cam_matrix[0][0] =  intr.fx #fx
depth_cam_matrix[0][2] =  intr.ppx #cx
depth_cam_matrix[1][1] =  intr.fy #fy
depth_cam_matrix[1][2] =  intr.ppy #cy
depth_cam_matrix[2][2] = 1

c_profile = profile.get_stream(rs.stream.color) # Fetch stream profile for depth stream
c_intr = c_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

color_cam_matrix = np.zeros((3,3), dtype=np.float)
color_cam_matrix[0][0] =  c_intr.fx #fx
color_cam_matrix[0][2] =  c_intr.ppx #cx
color_cam_matrix[1][1] =  c_intr.fy #fyq
color_cam_matrix[1][2] =  c_intr.ppy #cy
color_cam_matrix[2][2] =  1
color_distort = np.zeros(5, dtype=np.float)
color_distort[0] = c_intr.coeffs[0]  #k1
color_distort[1] = c_intr.coeffs[1]  #k2
color_distort[2] = c_intr.coeffs[2]  #p1
color_distort[3] = c_intr.coeffs[3]  #p2
color_distort[4] = c_intr.coeffs[4]  #k3

extrinsics = i_profile.get_extrinsics_to(c_profile)

Rt =  np.zeros((4,4), dtype=np.float32) #rotation matrix
Rt[0][0] = extrinsics.rotation[0]; Rt[0][1] = extrinsics.rotation[1];  Rt[0][2] = extrinsics.rotation[2]; Rt[0][3] = extrinsics.translation[0] 
Rt[1][0] = extrinsics.rotation[3]; Rt[1][1] = extrinsics.rotation[4];  Rt[1][2] = extrinsics.rotation[5]; Rt[1][3] = extrinsics.translation[1] 
Rt[2][0] = extrinsics.rotation[6]; Rt[2][1] = extrinsics.rotation[7];  Rt[2][2] = extrinsics.rotation[8]; Rt[2][3] = extrinsics.translation[2]
Rt[3][0] = 0; Rt[3][1] = 0;  Rt[3][2] = 0; Rt[3][3] = 1  
################# Variables for the RGBD OPENCV registration ###############################

##################### THREADING CONTROL VARIABLES ##################################

event = threading.Event()
orientation = 'h'  #horizontal or vertical orientation  
add = True         #add or substract the disparity     (left or right shift)
matrices = Matrices(mapx1_uv=mapx1_uv,mapy1_uv=mapy1_uv,mapx2_uv=mapx2_uv,mapy2_uv=mapy2_uv, cutMatrixUv=cutMatrixUv)

stream1 = cv2.cuda_Stream()
gpu1 = cuStream(stream1, resX, resY, matrices, base_rgb_uv, clipping_distance, orientation, add, event)
gpu1.start()

##################### THREADING CONTROL VARIABLES ##################################

# Streaming loop
try:
    
    while True:
        start = time.time_ns() 
        
        # ####################### INIT RS ALIGNED VERSION #############################
        
        # # Get frameset of color and depth
        # frame = rs_queue.wait_for_frame()

        # # Align the depth frame to color frame
        # aligned_frames = align.process(frame.as_frameset())

        # # Get aligned frames
        # aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        # color_frame = aligned_frames.get_color_frame()
        # uv_image = uCam.get_video_frame()  #uv camera

        # # Validate that both frames are valid
        # if not aligned_depth_frame or uv_image is None or not color_frame:
        #     continue

        # depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        # ####################### INIT RS ALIGNED VERSION #############################

        ####################### RAW VERSION #############################
        
        # Get frameset of color and depth
        frame = rs_queue.wait_for_frame()
        uv_image = uCam.get_video_frame()  #uv camera

        # Validate that both cameras return something 
        if not frame or uv_image is None:
            continue

        depth_frame = frame.as_frameset().get_depth_frame()
        color_frame = frame.as_frameset().get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        ####################### RAW VERSION #############################

        #Add the alpha chanel for the final alpha blending
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)  #TODO change for cv::cuda::cvtColor
        uv_image = cv2.cvtColor(uv_image, cv2.COLOR_RGB2RGBA)        #TODO change for cv::cuda::cvtColor

        ####################### RAW ALIGMENT OF THE RGB CAMERA USING OPENCV ###############################
        depth_image = cv2.rgbd.registerDepth( depth_cam_matrix, color_cam_matrix, color_distort, Rt, depth_image, (resX, resY), depthDilation=True )
        ####################### RAW ALIGMENT OF THE RGB CAMERA USING OPENCV ###############################

        gpu1.updateImages(color_image, uv_image, depth_image)
        event.set()

        elapsed = (time.time_ns() - start) / 1000000
        print ("CPU Time: " + str(elapsed))
        
        if gpu1.is_alive() == False:
            print('GPU1 thread is death')
            break
finally:
    print('Bye bye q ( n o n ) p ')
    uCam.stop()
    iCam.stop()
    pipeline.stop()

