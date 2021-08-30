############## STEREO CAMERA VERSION PROTOTYPE ###############################

#############################################################################
#    CUDA version for speed reasons only the UV camera is used               # 
#    the system works with the RS and the STEREO and LOW LIGHT cameras       #
#     ( matrixes: rsleft->rgb_right, right_rgb -> center ) IN THAT ORDER !!! # 
#    This version reciticates BOTH LEFT AND RIGHT CAMERAS !!!                #
#############################################################################

#TODO list
# fix the double images in the rectification   TODO
# fix or cut the rectification warp of the final images.   DONE
# optimize gpumat declarations and uploads           DONE 
# check if the camera taking photos creates a bottle neck  DONE
# remove salt and peper from the rectified images TODO
# TODO IMPORTANTE !!!! 
# TODO medir tiempos   (tiempo de camara mono 50 ms con una sola depth registration)
# TODO separar depth registration en otro hilo
# TODO subir matrices de las dos camaras an device 




import pyrealsense2 as rs
import numpy as np
import cv2
import time
from include.cameraThread import normalCamThread
from matplotlib import pyplot as plt  #matplotlib for DEBUG only
from include.cuda_stream27noThread import cuStream
from include.cuda_stream27noThread import Matrices


#PROGRAM CONSTANTS 
mapx1_right=None; mapx2_right=None; mapy1_right=None; mapy2_right=None
mapx1_left=None; mapx2_left=None; mapy1_left=None; mapy2_left=None
resX = 1280
resY = 720
Tdiv = 1000.0
Rtmul = 1.0 #1.429222
err = 0.0 #DEBUG
alphaValue = 1.0
fpsCount = 0

#This is a faster compositor  #FOR DEBUG ONLY
def fastAlphaBlend(fg,bg,alpha):
    # a = alpha[:, :, np.newaxis]
    blended = cv2.convertScaleAbs(fg * alpha + bg * (1-alpha))
    return blended

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
    vRoi = []
    vRoi.append( max(vroi1[0][0], vroi2[0][0]) )  #x
    vRoi.append( max(vroi1[1][0], vroi2[1][0]) )  #y
    vRoi.append( min(vroi1[2][0], vroi2[2][0]) )  #w
    vRoi.append( min(vroi1[3][0], vroi2[3][0]) )  #h
 
    fint.release()
    fext.release()
    return [M1, M2, D1, D2, R1, R2, P1, P2, T, R, vRoi]

#calculates the rectify matrices from the camera parameters
# def undistortUV(M1, M2, D1, D2, R1, R2, P1, P2):
#     # PROBAR ESTA OPCION !!! OPCIONAL 1!!
#     # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
#     # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
#     #Mat rmap[2][2];
#     global mapx1_right; global mapx2_right; global mapy1_right; global mapy2_right
#     mapx1_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
#     mapy1_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
#     mapx2_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
#     mapy2_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')

#     mapx1_uv, mapy1_uv = cv2.initUndistortRectifyMap(M1, D1, R1, P1,(resX, resY), cv2.CV_32F)  #original cv2.CV_16SC2 
#     mapx2_uv, mapy2_uv = cv2.initUndistortRectifyMap(M2, D2, R2, P2,(resX, resY), cv2.CV_32F)


#interactive matlab plot
plt.ion() 

#Objects to get the camera frames in different thread 
rCam = normalCamThread(resX, resY, 4, 90)  #right rgb stereo camera
cCam = normalCamThread(resX, resY, 5, 90)  #center camera (UV / INFRA sensor)
lCam = normalCamThread(resX, resY, 3, 90)  #left rgb stereo camera
rCam.start()
cCam.start()
lCam.start()

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

# read the matrices from the files 
# matrices[M1, M2, D1, D2, R1, R2, P1, P2, T, R, vRoi]
RC_Mats = loadCamFiles('./cam_params/right_center_extrinsics.yml', './cam_params/right_center_intrinsics.yml')  
LC_Mats = loadCamFiles('./cam_params/left_center_extrinsics.yml', './cam_params/left_center_intrinsics.yml')  
# CALCULATE RIGHT CAMERA DISTORITON MATRICES
mapx1_right = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapy1_right = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapx2_right = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapy2_right = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapx1_right, mapy1_right = cv2.initUndistortRectifyMap(RC_Mats[0], RC_Mats[2], RC_Mats[4], RC_Mats[6], (resX, resY), cv2.CV_32F)  
mapx2_right, mapy2_right = cv2.initUndistortRectifyMap(RC_Mats[1], RC_Mats[3], RC_Mats[5], RC_Mats[7], (resX, resY), cv2.CV_32F)
# CALCULATE LEFT CAMERA DISTORTION MATRICES 
mapx1_left = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapy1_left = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapx2_left = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapy2_left = np.ndarray(shape=(resY, resX, 1), dtype='float32')
mapx1_left, mapy1_left = cv2.initUndistortRectifyMap(LC_Mats[0], LC_Mats[2], LC_Mats[4], LC_Mats[6], (resX, resY), cv2.CV_32F)  
mapx2_left, mapy2_left = cv2.initUndistortRectifyMap(LC_Mats[1], LC_Mats[3], LC_Mats[5], LC_Mats[7], (resX, resY), cv2.CV_32F)
# CALCULATE TEH BASE DISTANCE BETWEEN THE SENSOR AND THE CAMERAS 
# base_rgb_right = abs(float(RC_Mats[8][0])) * abs(float(RC_Mats[0][0][0])) #RIGHT RGB BASE DISTANCE
base_rgb_right = abs(float(RC_Mats[8][0])) * abs(float(RC_Mats[0][0][0]))
base_rgb_left =  abs(float(LC_Mats[8][0])) * abs(float(LC_Mats[0][0][0])) #LEFT RGB BASE DISTANCE

# undistortUV(RC_Mats[0], RC_Mats[1], RC_Mats[2], RC_Mats[3], RC_Mats[4], RC_Mats[5], RC_Mats[6], RC_Mats[7])  # calculate the distortion matrices
# base_rgb_right = abs(float(RC_Mats[8][0])) * abs(float(RC_Mats[0][0][0]))                     # T[1] M1[fx] cause rectified on Y (horizontal cameras)

# CALCULATE THE MTRIX TO CUT THE FINAL IMAGE (RIGHT)
# print ("VRoi dimentions: " + str( RC_Mats[10]))
pts1 = np.float32([[RC_Mats[10][0],RC_Mats[10][1]], [RC_Mats[10][0]+RC_Mats[10][2], RC_Mats[10][1] ], [RC_Mats[10][0], RC_Mats[10][1]+RC_Mats[10][3]], [RC_Mats[10][0]+RC_Mats[10][2], RC_Mats[10][1]+RC_Mats[10][3]] ])
pts2 = np.float32([[0,0],[resX,0],[0,resY],[resX,resY]])
cutMatrixRight = cv2.getPerspectiveTransform(pts1,pts2)

# CALCULATE THE MTRIX TO CUT THE FINAL IMAGE (LEFT)
# print ("VRoi dimentions: " + str( LC_Mats[10]))
pts1 = np.float32([[LC_Mats[10][0],LC_Mats[10][1]], [LC_Mats[10][0]+LC_Mats[10][2], LC_Mats[10][1] ], [LC_Mats[10][0], LC_Mats[10][1]+LC_Mats[10][3]], [LC_Mats[10][0]+LC_Mats[10][2], LC_Mats[10][1]+LC_Mats[10][3]] ])
pts2 = np.float32([[0,0],[resX,0],[0,resY],[resX,resY]])
cutMatrixLeft = cv2.getPerspectiveTransform(pts1,pts2)

################# Variables for the RGBD OPENCV registration ###############################
i_profile = profile.get_stream(rs.stream.depth)
intr = i_profile.as_video_stream_profile().get_intrinsics()

# Mats[M1, M2, D1, D2, R1, R2, P1, P2, T, R]  
depth_cam_matrix = np.zeros((3,3), dtype=np.float)
depth_cam_matrix[0][0] =  intr.fx #fx
depth_cam_matrix[0][2] =  intr.ppx #cx
depth_cam_matrix[1][1] =  intr.fy #fy
depth_cam_matrix[1][2] =  intr.ppy #cy
depth_cam_matrix[2][2] = 1
# depth_cam_matrix = RSL_R_Mats[0]

# c_profile = profile.get_stream(rs.stream.color) # Fetch stream profile for depth stream
# c_intr = c_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

RSL_R_Mats = loadCamFiles('./cam_params/rsleft_right_extrinsics.yml', './cam_params/rsleft_right_intrinsics.yml')
right_color_cam_matrix = RSL_R_Mats[1]  #center to right M2
right_color_distort = RSL_R_Mats[3]     #center to right D2
R = RSL_R_Mats[9]
T = RSL_R_Mats[8]

Rt_right =  np.zeros((4,4), dtype=np.float32) #rotation matrix
Rt_right[0][0] = R[0][0]*Rtmul; Rt_right[0][1] = R[0][1]*Rtmul;  Rt_right[0][2] = R[0][2]*Rtmul; Rt_right[0][3] = T[0]/Tdiv
Rt_right[1][0] = R[1][0]*Rtmul; Rt_right[1][1] = R[1][1]*Rtmul;  Rt_right[1][2] = R[1][2]*Rtmul; Rt_right[1][3] = T[1]/Tdiv
Rt_right[2][0] = R[2][0]*Rtmul; Rt_right[2][1] = R[2][1]*Rtmul;  Rt_right[2][2] = R[2][2]*Rtmul; Rt_right[2][3] = T[2]/Tdiv
Rt_right[3][0] = 0; Rt_right[3][1] = 0;  Rt_right[3][2] = 0; Rt_right[3][3] = 1  

RSL_L_Mats = loadCamFiles('./cam_params/rsleft_left_extrinsics.yml', './cam_params/rsleft_left_intrinsics.yml')
left_color_cam_matrix = RSL_L_Mats[1]  #center to left M2
left_color_distort = RSL_L_Mats[3]     #center to left D2
R = RSL_L_Mats[9]
T = RSL_L_Mats[8]

Rt_left =  np.zeros((4,4), dtype=np.float32) #rotation matrix
Rt_left[0][0] = R[0][0]*Rtmul; Rt_left[0][1] = R[0][1]*Rtmul;  Rt_left[0][2] = R[0][2]*Rtmul; Rt_left[0][3] = T[0]/Tdiv
Rt_left[1][0] = R[1][0]*Rtmul; Rt_left[1][1] = R[1][1]*Rtmul;  Rt_left[1][2] = R[1][2]*Rtmul; Rt_left[1][3] = T[1]/Tdiv
Rt_left[2][0] = R[2][0]*Rtmul; Rt_left[2][1] = R[2][1]*Rtmul;  Rt_left[2][2] = R[2][2]*Rtmul; Rt_left[2][3] = T[2]/Tdiv
Rt_left[3][0] = 0; Rt_left[3][1] = 0;  Rt_left[3][2] = 0; Rt_left[3][3] = 1  

################# Variables for the RGBD OPENCV registration ###############################

##################### THREADING CONTROL VARIABLES ##################################

# event_right = threading.Event()
# matrices_right = Matrices(mapx1_uv=mapx1_right,mapy1_uv=mapy1_right,mapx2_uv=mapx2_right,mapy2_uv=mapy2_right, cutMatrixUv=cutMatrixRight)
# stream_right = cv2.cuda_Stream() 
# gpu_right = cuStream(stream_right, resX, resY, matrices_right, base_rgb_right, clipping_distance, 'h', True, event_right, 'right')
# gpu_right.start()


# event_left = threading.Event()
# matrices_left = Matrices(mapx1_uv=mapx1_left,mapy1_uv=mapy1_left,mapx2_uv=mapx2_left,mapy2_uv=mapy2_left, cutMatrixUv=cutMatrixLeft)
# stream_left = cv2.cuda_Stream()
# gpu_left = cuStream(stream_left, resX, resY, matrices_left, base_rgb_left, clipping_distance, 'h', False, event_left, 'left')
# gpu_left.start()
##################### THREADING CONTROL VARIABLES ##################################

##################### NO THREAD STREAM VERSION ##################################

matrices_right = Matrices(mapx1_uv=mapx1_right,mapy1_uv=mapy1_right,mapx2_uv=mapx2_right,mapy2_uv=mapy2_right, cutMatrixUv=cutMatrixRight)
stream_right = cv2.cuda_Stream() 
gpu_right = cuStream(stream_right, resX, resY, matrices_right, base_rgb_right, clipping_distance, 'h', True, 'right')

matrices_left = Matrices(mapx1_uv=mapx1_left,mapy1_uv=mapy1_left,mapx2_uv=mapx2_left,mapy2_uv=mapy2_left, cutMatrixUv=cutMatrixLeft)
stream_left = cv2.cuda_Stream()
gpu_left = cuStream(stream_left, resX, resY, matrices_left, base_rgb_left, clipping_distance, 'h', False, 'left')

##################### NO THREAD STREAM VERSION ##################################

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
        color_image_left = lCam.get_video_frame()  #rgb left camera
        sensor_image = cCam.get_video_frame()  #sensor image (uv or infrared camera)
        color_image_right = rCam.get_video_frame()  #rgb right camera

        # Validate that both cameras return something 
        if not frame or sensor_image is None or color_image_left is None or color_image_right is None:
            continue

        depth_frame = frame.as_frameset().get_depth_frame()
        # color_frame = frame.as_frameset().get_color_frame()
        depth_image_left = np.asanyarray(depth_frame.get_data())
        depth_image_right = depth_image_left.copy()
        # color_image = np.asanyarray(color_frame.get_data())

        ####################### RAW VERSION #############################

        #Add the alpha chanel for the final alpha blending
        color_image_left = cv2.cvtColor(color_image_left, cv2.COLOR_RGB2RGBA)  #TODO change for cv::cuda::cvtColor
        color_image_right = cv2.cvtColor(color_image_right, cv2.COLOR_RGB2RGBA)  #TODO change for cv::cuda::cvtColor
        sensor_image_left = cv2.cvtColor(sensor_image, cv2.COLOR_RGB2RGBA)        #TODO change for cv::cuda::cvtColor
        sensor_image_right = sensor_image_left.copy()
        ####################### RAW ALIGMENT OF THE RGB CAMERA USING OPENCV ###############################
        depth_image_left  = cv2.rgbd.registerDepth( depth_cam_matrix, left_color_cam_matrix, left_color_distort, Rt_left, depth_image_left, (resX, resY), depthDilation=True )
        depth_image_right = cv2.rgbd.registerDepth( depth_cam_matrix, right_color_cam_matrix, right_color_distort, Rt_right, depth_image_right, (resX, resY), depthDilation=True )
        ####################### RAW ALIGMENT OF THE RGB CAMERA USING OPENCV ###############################

        ##### DEPTH MASK RECTIFICATION DEBUG #####################################
        # alphax = np.full((resY, resX,4), 0.5, dtype="float32")   #DEBUG
        # debugito = np.dstack((depth_image_left,depth_image_left,depth_image_left,depth_image_left))    #DEBUG
        # # debugito = np.dstack((depth_image,depth_image,depth_image,depth_image))   #DEBUG
        # pinchi_left = fastAlphaBlend(color_image_left,debugito, alphax)  #DEBUG
        # cv2.imshow('blend debug left', pinchi_left)  #DEBUG
        # key = cv2.waitKey(1)  #DEBUG

        # # alphax = np.full((resY, resX,4), 0.5, dtype="float32")   #DEBUG
        # debugito = np.dstack((depth_image_right,depth_image_right,depth_image_right,depth_image_right))    #DEBUG
        # # debugito = np.dstack((depth_image,depth_image,depth_image,depth_image))   #DEBUG
        # pinchi_right = fastAlphaBlend(color_image_right,debugito, alphax)  #DEBUG
        # cv2.imshow('blend debug right', pinchi_right)  #DEBUG
        # key = cv2.waitKey(1)  #DEBUG
         ##### DEPTH MASK RECTIFICATION DEBUG #####################################
        elapsed = (time.time_ns() - start) / 1000000
        start2 = time.time_ns() 

        gpu_left.updateImages(color_image_left, sensor_image_left, depth_image_left)   #left left OK
        left_return = gpu_left.run()
        gpu_right.updateImages(color_image_right, sensor_image_right, depth_image_right)  #right right BAD
        right_return = gpu_right.run()

        # stream_left.waitForCompletion()
        # stream_right.waitForCompletion()
        

        
        elapsed2 = (time.time_ns() - start2) / 1000000
        print ("CPU Time: " + str(elapsed))
        print ("GPU Time: " + str(elapsed2))
        
        if right_return == -1:
            print('GPU Right thread is death')
            break
        if left_return == -1:
            print('GPU Left thread is death')
            break
finally:
    print('Bye bye q ( n o n ) p ')
    rCam.stop()
    cCam.stop()
    lCam.stop()
    pipeline.stop()

