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
from cameraThread import normalCamThread
from matplotlib import pyplot as plt  #matplotlib for DEBUG only

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

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# baseline, F, matrices[M1, M2, D1, D2, R1, R2, P1, P2]
Mats = loadCamFiles('./cam_params/rgb_uv_extrinsics.yml', './cam_params/rgb_uv_intrinsics.yml')  
undistortUV(Mats[0], Mats[1], Mats[2], Mats[3], Mats[4], Mats[5], Mats[6], Mats[7])  # calculate the distortion matrices
base_rgb_uv = abs(float(Mats[8][0])) * abs(float(Mats[0][0][0]))                     # T[1] M1[fx] cause rectified on Y (horizontal cameras)

# CALCULATE THE MATRIX TO CUT THE FINAL IMAGE
# print ("VRoi dimentions: " + str(uvVroi))
pts1 = np.float32([[uvVroi[0],uvVroi[1]], [uvVroi[0]+uvVroi[2], uvVroi[1] ], [uvVroi[0], uvVroi[1]+uvVroi[3]], [uvVroi[0]+uvVroi[2], uvVroi[1]+uvVroi[3]] ])
pts2 = np.float32([[0,0],[resX,0],[0,resY],[resX,resY]])
cutMatrixUv = cv2.getPerspectiveTransform(pts1,pts2)

# GPU FILTERS INIT (for mask creation)
eSize = 10
e_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*eSize + 1, 2*eSize+1), (eSize, eSize))
cuErosionFilter = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, e_element)
dSize = 15
d_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dSize + 1, 2*dSize+1), (dSize, dSize))
cuDilateFilter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, d_element)
gSize = 15
cuGaussFilter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (gSize, gSize), 0, 0)

# Filters for salt and pepper remove (after registration remap)
# cuResultSplit = cv2.cuda_GpuMat()
# sp_e_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))  # 3x3 size
# saltPepperErotion = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8U, sp_e_element)  
# sp_d_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3 , 3), (-1, -1))
# saltPepperDilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8U, sp_d_element)
cuMedianFilter = cv2.cuda.createMedianFilter(cv2.CV_8UC1, 3)	

#Depth to Disparity block 
#control variables
orientation = 'h'  #horizontal or vertical orientation  
add = True         #add or substract the disparity     (left or right shift)

# GPU MATRICES INIT
cuDepth = cv2.cuda_GpuMat() 
cuUv = cv2.cuda_GpuMat(); 
cuColor = cv2.cuda_GpuMat()
cuDisp = cv2.cuda_GpuMat(); cuDisp.upload(np.full((resY, resX), 0, dtype="float32"))  #this static init may cause a bug
cuMapx1_uv = cv2.cuda_GpuMat(); cuMapx1_uv.upload(mapx1_uv)  #these are static
cuMapy1_uv = cv2.cuda_GpuMat(); cuMapy1_uv.upload(mapy1_uv)  #these are static
cuMapx2_uv = cv2.cuda_GpuMat();  cuMapx2_uv.upload(mapx2_uv)  #these are static
cuMapy2_uv = cv2.cuda_GpuMat();  cuMapy2_uv.upload(mapy2_uv)  #these are static

ones_f = cv2.cuda_GpuMat();  ones_f.upload(np.full((resY, resX), 1.0, dtype="float32"))          #static
bases = cv2.cuda_GpuMat();  bases.upload(np.full((resY, resX), base_rgb_uv, dtype="float32"))    #static

grid = np.indices((resY, resX))                                                   #static
cuGrid_x = cv2.cuda_GpuMat();  cuGrid_x.upload(grid[1].astype(np.float32))        #static
cuGrid_y = cv2.cuda_GpuMat();  cuGrid_y.upload(grid[0].astype(np.float32))        #static


# Rectify and remap block 
cuRecColor = cv2.cuda_GpuMat(); cuRecColor.upload(np.full((resY, resX), 0, dtype="float32"))
cuRecUv = cv2.cuda_GpuMat();    cuRecUv.upload(np.full((resY, resX, 4), 0, dtype="float32"))
cuRecMask = cv2.cuda_GpuMat();  cuRecMask.upload(np.full((resY, resX), 0, dtype="float32"))
cuResult = cv2.cuda_GpuMat();  cuResult.upload(np.full((resY, resX, 4), 0, dtype="float32"))
cuFinal = cv2.cuda_GpuMat();  cuFinal.upload( np.full (  (resY, resX, 4), 0, dtype="float32"))
ones = cv2.cuda_GpuMat();  ones.upload(np.full((resY, resX, 3), 1, dtype="float32") )

cuMask = cv2.cuda_GpuMat(); cuMask.upload(np.full((resY, resX), 0, dtype="uint16") )
cuClip1 = cv2.cuda_GpuMat();  cuClip1.upload(np.full((resY, resX), 0, dtype="uint8") )
cuClip2 = cv2.cuda_GpuMat();  cuClip2.upload(np.full((resY, resX), 0, dtype="uint8") )
cuMaskDist = cv2.cuda_GpuMat(); cuMaskDist.upload(np.full((resY, resX), int(clipping_distance), dtype="uint16") )
cuMaskZeros = cv2.cuda_GpuMat(); cuMaskZeros.upload(np.full((resY, resX), 0, dtype="uint16") )
# maskZeros = cv2.cuda_GpuMat(); maskZeros.upload(np.full((resY, resX), 0, dtype="uint8"))


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

# Streaming loop
try:
    
    while True:
        start = time.time_ns() 
        
        ####################### INIT SECTION it takes 4 ms  #############################
        
        # Get frameset of color and depth
        frame = rs_queue.wait_for_frame()

        # Align the depth frame to color frame
        aligned_frames = align.process(frame.as_frameset())

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        uv_image = uCam.get_video_frame()  #uv camera

        # Validate that both frames are valid
        if not aligned_depth_frame or uv_image is None or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #Add the alpha chanell for the final alpha blending
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)  #TODO change for cv::cuda::cvtColor
        uv_image = cv2.cvtColor(uv_image, cv2.COLOR_RGB2RGBA)        #TODO change for cv::cuda::cvtColor

        #only for DEBUG
        # uv_image = cv2.cvtColor(uv_image, cv2.COLOR_BGR2GRAY)
        # uv_image = cv2.applyColorMap(uv_image, cv2.COLORMAP_HOT)

        # uv_image = uCam.get_video_frame()  #uv camera
        # frame = rs_queue.wait_for_frame()
        # if not frame or uv_image is None:
        #     continue

        # depth_frame = frame.as_frameset().get_depth_frame()
        # color_frame = frame.as_frameset().get_color_frame()
        # depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        # uv_image = cv2.cvtColor(uv_image, cv2.COLOR_BGR2GRAY)
        # uv_image = cv2.applyColorMap(uv_image, cv2.COLORMAP_HOT)
        
        
        ####################### INIT SECTION it takes 4 ms  #############################

        ####################### RAW ALIGMENT OF THE RGB CAMERA USING OPENCV ###############################
        # regDepth = cv.rgbd.registerDepth( depth_cam_matrix, color_cam_matrix, color_distort, Rt, depth_image, (c_intr.width, c_intr.height), depthDilation=False)s
        # depth_image = cv2.rgbd.registerDepth( depth_cam_matrix, color_cam_matrix, color_distort, Rt, depth_image, (resX, resY), depthDilation=True )
        ####################### RAW ALIGMENT OF THE RGB CAMERA USING OPENCV ###############################
        # ##################### Remove background  GPU ################################  This steqp takes 17ms to 20ms too
        # start2 = time.time_ns()        
        # mask = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0.0, alphaValue)   # I can optimize this....  cv::cuda::threshold	
        # cuMask.upload(mask.astype(np.float32))

        # # cuMask = cuErosionFilter.apply(cuMask)
        # cuMask = cuDilateFilter.apply(cuMask)
        # cuMask = cuGaussFilter.apply(cuMask)
        # elapsed2 = (time.time_ns() - start2) / 1000000
        # ##################### Remove background  GPU ################################ 

        # ##################### Remove background  GPU ################################  This steqp takes 17ms to 20ms too
        start2 = time.time_ns()   
        # mask = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0.0, alphaValue)   # orignal formula	
        cuMask.upload(np.array(depth_image))
        cuClip1 = cv2.cuda.compare(cuMask, cuMaskDist, cv2.CMP_LE)   #depth < clipping_distance (inverted)
        cuClip2 = cv2.cuda.compare(cuMask, cuMaskZeros, cv2.CMP_GT)  #depth >= 0   (inverted)
        cuMask = cv2.cuda.bitwise_and(cuClip1, cuClip2)   #bitwise OR
        
        # cuMask = cuErosionFilter.apply(cuMask)
        cuMask = cuDilateFilter.apply(cuMask)
        cuMask = cuGaussFilter.apply(cuMask)
        elapsed2 = (time.time_ns() - start2) / 1000000
        # ##################### Remove background  GPU ################################ 
        

        ############### GET DISPARITY GPU for UV CAMERA ########################################### This step takes 5ms 
        
        # Make all the uploads necessary (depth, mapx) 
        cuDepth.upload(depth_image.astype(np.float32))  #CHECK THIS, THE CAST WAS ORIGINALLY AFTER THE REMAP

        #Remap of the depth map
        cuDisp = cv2.cuda.remap(cuDepth, cuMapx1_uv, cuMapy1_uv, cv2.INTER_LINEAR)
        cuDisp = cv2.cuda.divide(ones_f, cuDisp)
        cuDisp = cv2.cuda.multiply(cuDisp, bases)

        #Transform the depth map to disparity map aligned to the external camera 
        if orientation == 'h':
            if add:
                cuDisp = cv2.cuda.add(cuGrid_x, cuDisp)
            else:
                cuDisp = cv2.cuda.subtract(cuGrid_x, cuDisp)
        elif orientation == 'v':
            if add:
                cuDisp = cv2.cuda.add(cuGrid_y, cuDisp)
            else:
                cuDisp = cv2.cuda.subtract(cuGrid_y, cuDisp)
        
        ############### GET DISPARITY GPU for UV CAMERA ###########################################


        ############### Rectify the UV, RGB and Mask images ########################################### Both rectify and remap sections take around 5ms
        
        cuUv.upload(uv_image)
        cuColor.upload(color_image)
        
        cuRecColor = cv2.cuda.remap(cuColor, cuMapx1_uv, cuMapy1_uv, cv2.INTER_LINEAR)  #rgb
        cuRecUv = cv2.cuda.remap(cuUv, cuMapx2_uv, cuMapy2_uv, cv2.INTER_LINEAR) #uv
        cuRecMask = cv2.cuda.remap(cuMask, cuMapx1_uv, cuMapy1_uv, cv2.INTER_LINEAR) #mask aligned to rgb
        
        ###############   Make the remaping of the UV image over the RGB image using the computed disparity Map
        if orientation == 'h':
            cuResult = cv2.cuda.remap(cuRecUv, cuDisp, cuGrid_y, cv2.INTER_LINEAR)  #returns a uint8
        elif orientation == 'v':
            cuResult = cv2.cuda.remap(cuRecUv, cuGrid_x, cuDisp, cv2.INTER_LINEAR)  #returns a uint8
        
        ############### Rectify the UV, RGB and Mask images ########################################### Both rectify and remap sections take around 5ms


        ###############   Mix the MASK and the rectified RGB and UV images ######################## #This section takes 6 to 10 ms (float conversion 25ms)
        recMask = cuRecMask.download()
        resu =  cuResult.download()
        resu[:, :, 3] = cv2.convertScaleAbs(recMask, alpha=255)
        cuResult.upload(resu)
        
        cuFinal = cv2.cuda.alphaComp(cuResult, cuRecColor, cv2.cuda.ALPHA_OVER )

        ###############   Mix the MASK and the rectified RGB and UV images ######################## #This section takes 6 to 10 ms (with floats 25ms)

        ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
        cuFinal = cv2.cuda.warpPerspective(cuFinal, cutMatrixUv, (resX,resY))
        ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
            
        # final = cv2.convertScaleAbs(cuFinal.download())
        final = cuFinal.download()
        cv2.imshow('final', final)

        # mask = cuMask.download() 
        end = time.time_ns() 
        elapsed = (end - start) / 1000000
        print ("Elapsed: " + str(elapsed))
        print ("Elapsed2: " + str(elapsed2))

        # fpsCount = fpsCount +1
        # if ( ((end - start) / 1000000) > 1000):
        #     start = time.time_ns() 
        #     print ('FPS:'+ str(fpsCount))
        #     fpsCount = 0

        # plt.imshow(mask)
        # plt.show()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
finally:
    uCam.stop()
    iCam.stop()
    pipeline.stop()

