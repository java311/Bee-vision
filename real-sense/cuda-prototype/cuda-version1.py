import pyrealsense2 as rs
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt  #matplotlib for DEBUG only

#PROGRAM CONSTANTS 
mapx1_uv=None; mapx2_uv=None; mapy1_uv=None; mapy2_uv=None
resX = 1280
resY = 720
err = 0.0 #DEBUG

#This is a faster compositor
def fastAlphaBlend(fg,bg,alpha):
    # MY VERSION
    a = alpha[:, :, np.newaxis]
    blended = cv2.convertScaleAbs(fg * a + bg * (1-a))
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
 
    fint.release()
    fext.release()
    return [M1, M2, D1, D2, R1, R2, P1, P2, T]

#calculates the rectify matrices from the camera parameters
def undistortUV(M1, M2, D1, D2, R1, R2, P1, P2):
    # PROBAR ESTA OPCION !!! OPCIONAL 1!!
    # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    #Mat rmap[2][2];
    global mapx1_uv; global mapx2_uv; global mapy1_uv; global mapy2_uv;
    mapx1_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy1_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapx2_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy2_uv = np.ndarray(shape=(resY, resX, 1), dtype='float32')

    mapx1_uv, mapy1_uv = cv2.initUndistortRectifyMap(M1, D1, R1, P1,(resX, resY), cv2.CV_32F)  #original cv2.CV_16SC2 
    mapx2_uv, mapy2_uv = cv2.initUndistortRectifyMap(M2, D2, R2, P2,(resX, resY), cv2.CV_32F)


# def getDisparity(depthMap, orientation, base, mapx, mapy, positive, err):  #horizontal
#     #gpu variables 
#     cuDepth = cv2.cuda_GpuMat(); cuDepth.upload(depthMap.astype(np.float32))  #CHECK THIS, THE CAST WAS ORIGINALLY AFTER THE REMAP
#     cuMapx = cv2.cuda_GpuMat();  cuMapx.upload(mapx)
#     cuMapy = cv2.cuda_GpuMat();  cuMapy.upload(mapy)

#     #first remap the depthMap
#     cuDisp = cv2.cuda_GpuMat();  cuDisp.upload(np.full_like(depthMap, 0, dtype="float32")) 
#     cuDisp = cv2.cuda.remap(cuDepth, cuMapx, cuMapy, cv2.INTER_LINEAR)

#     ones = cv2.cuda_GpuMat();  ones.upload(np.full_like(depthMap, 1, dtype="float32"))
#     bases = cv2.cuda_GpuMat();  bases.upload(np.full_like(depthMap, base, dtype="float32")) 

#     cuDisp = cv2.cuda.divide(ones, cuDisp)
#     cuDisp = cv2.cuda.multiply(cuDisp, bases)

#     grid = np.indices((resY, resX))
#     cuGrid_x = cv2.cuda_GpuMat();  cuGrid_x.upload(grid[1].astype(np.float32))
#     cuGrid_y = cv2.cuda_GpuMat();  cuGrid_y.upload(grid[0].astype(np.float32))

#     if orientation == 'h':
#         if positive:
#             cuDisp = cv2.cuda.add(cuGrid_x, cuDisp)
#         else:
#             cuDisp = cv2.cuda.subtract(cuGrid_x, cuDisp)




#interactive matlab plot
# plt.ion() 

# Create a pipeline
pipeline = rs.pipeline()
cap = cv2.VideoCapture(4)   #chinese with leds     (infra)
cap2 = cv2.VideoCapture(1)  #chinese with no leds  (uvcam)

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, resX, resY, rs.format.z16, 30)  #GETS BETTER DEPTH READINGS !!
config.enable_stream(rs.stream.color, resX, resY, rs.format.bgr8, 30)  #GETS BETTER DEPTH READINGS !!
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

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

# Check if the UV camera opened successfully
if (cap.isOpened() == False or cap2.isOpened() == False):
    print("Error opening video stream or file")

#change webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resX)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,resY)  
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, resX)  
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,resY)  

# baseline, F, matrices[M1, M2, D1, D2, R1, R2, P1, P2]
Mats = loadCamFiles('./cam_params/rgb_uv_extrinsics.yml', './cam_params/rgb_uv_intrinsics.yml')  
undistortUV(Mats[0], Mats[1], Mats[2], Mats[3], Mats[4], Mats[5], Mats[6], Mats[7])  # calculate the distortion matrices
base_rgb_uv = abs(float(Mats[8][0])) * abs(float(Mats[0][0][0]))                     # T[1] M1[fx] cause rectified on Y (horizontal cameras)


# Streaming loop
try:
    while True:
        # print ("loop")
        start = time.time_ns() 

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        ret, uv_image = cap2.read()  #uv camera
        ret2, infra_image = cap.read()  #infra camera

        uv_image = cv2.cvtColor(uv_image, cv2.COLOR_BGR2GRAY)
        uv_image = cv2.applyColorMap(uv_image, cv2.COLORMAP_HOT)

        # Validate that both frames are valid
        if not aligned_depth_frame or ret == False or ret2 == False or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        ##################### Remove background  GPU ################################ 
        mask = np.where((depth_image > clipping_distance) | (depth_image <= 0.0), 0.0, 1.0)   # I can optimize this....         
        cuMask = cv2.cuda_GpuMat()
        cuMask.upload(mask.astype(np.float32))

        eSize = 10
        e_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*eSize + 1, 2*eSize+1), (eSize, eSize))
        cuErosionFilter = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_32F, e_element)
        dSize = 15
        d_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dSize + 1, 2*dSize+1), (dSize, dSize))
        cuDilateFilter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_32F, d_element)
        gSize = 15
        cuGaussFilter = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (gSize, gSize), 0, 0)
        cuMask = cuErosionFilter.apply(cuMask)
        cuMask = cuDilateFilter.apply(cuMask)
        cuMask = cuGaussFilter.apply(cuMask)
        ##################### Remove background  GPU ################################ 
        
       ############### GET DISPARITY GPU for UV CAMERA ###########################################
        # disp_x, disp_y = getDisparity(depth_image, 'h', base_rgb_uv, mapx1_uv, mapy1_uv, True, 0.0)  #Transforms the depth map to disparity map 
        #control variables
        orientation = 'h'  #horizontal or vertical orientation
        add = True

        #gpu variables 
        cuDepth = cv2.cuda_GpuMat(); cuDepth.upload(depth_image.astype(np.float32))  #CHECK THIS, THE CAST WAS ORIGINALLY AFTER THE REMAP
        cuMapx1_uv = cv2.cuda_GpuMat();  cuMapx1_uv.upload(mapx1_uv)
        cuMapy1_uv = cv2.cuda_GpuMat();  cuMapy1_uv.upload(mapy1_uv)

        ###############  Remap of the depth map
        cuDisp = cv2.cuda_GpuMat();  cuDisp.upload(np.full_like(depth_image, 0, dtype="float32")) 
        cuDisp = cv2.cuda.remap(cuDepth, cuMapx1_uv, cuMapy1_uv, cv2.INTER_LINEAR)

        ones = cv2.cuda_GpuMat();   ones.upload(np.full_like(depth_image, 1, dtype="float32"))
        bases = cv2.cuda_GpuMat();  bases.upload(np.full_like(depth_image, base_rgb_uv, dtype="float32")) 

        cuDisp = cv2.cuda.divide(ones, cuDisp)
        cuDisp = cv2.cuda.multiply(cuDisp, bases)

        grid = np.indices((resY, resX))
        cuGrid_x = cv2.cuda_GpuMat();  cuGrid_x.upload(grid[1].astype(np.float32))
        cuGrid_y = cv2.cuda_GpuMat();  cuGrid_y.upload(grid[0].astype(np.float32))

        ###############  Transform the depth map to disparity map aligned to the external camera 
        if orientation == 'h':
            if add:
                cuDisp = cv2.cuda.add(cuGrid_x, cuDisp)
            else:
                cuDisp = cv2.cuda.subtract(cuGrid_x, cuDisp)
            # cuDispX = cuDisp
            # cuDispY = cuGrid_y
        elif orientation == 'v':
            if add:
                cuDisp = cv2.cuda.add(cuGrid_y, cuDisp)
            else:
                cuDisp = cv2.cuda.subtract(cuGrid_y, cuDisp)
            # cuDispX = cuGrid_x
            # cuDispY = cuDisp
        
        # rec_color_image, rec_uv_image, rec_mask = remap(color_image, uv_image, mask, mapx1_uv, mapy1_uv, mapx2_uv, mapy2_uv)  #Rectify UV, RGB and MASK
        # def remap(left, right, mask, mapx1, mapy1, mapx2, mapy2):  #left is rgb_RS, right is UV, depth is UV depth 

        ############### Rectify the UV, RGB and Mask images 
        cuUv = cv2.cuda_GpuMat();  cuUv.upload(uv_image)
        cuColor = cv2.cuda_GpuMat();  cuColor.upload(color_image)
        cuMapx2_uv = cv2.cuda_GpuMat();  cuMapx2_uv.upload(mapx2_uv)
        cuMapy2_uv = cv2.cuda_GpuMat();  cuMapy2_uv.upload(mapy2_uv)

        cuRecColor = cv2.cuda_GpuMat(); cuRecColor.upload(np.full_like(color_image, 0, dtype="uint8"))
        cuRecUv = cv2.cuda_GpuMat();    cuRecUv.upload(np.full_like(uv_image, 0, dtype="uint8"))
        cuRecMask = cv2.cuda_GpuMat();  cuRecMask.upload(np.full_like(mask, 0, dtype="float32"))

        cuRecColor = cv2.cuda.remap(cuColor, cuMapx1_uv, cuMapy1_uv, cv2.INTER_LINEAR)  #rgb
        cuRecUv = cv2.cuda.remap(cuUv, cuMapx2_uv, cuMapy2_uv, cv2.INTER_LINEAR) #uv
        cuRecMask = cv2.cuda.remap(cuMask, cuMapx1_uv, cuMapy1_uv, cv2.INTER_LINEAR) #mask aligned to rgb
        
        ###############   Make the remaping of the UV image over the RGB image using the computed disparity Map
        cuResult = cv2.cuda_GpuMat();  cuResult.upload(np.full_like(uv_image, 0, dtype="uint8"))
        if orientation == 'h':
            cuResult = cv2.cuda.remap(cuRecUv, cuDisp, cuGrid_y, cv2.INTER_LINEAR)
        elif orientation == 'v':
            cuResult = cv2.cuda.remap(cuRecUv, cuGrid_x, cuDisp, cv2.INTER_LINEAR)

        ###############   Mix the MASK and the rectified RGB and UV images 

        # cuTmp1 = cv2.cuda_GpuMat()
        # cuTmp1 = cv2.cuda.multiply(cuRecUv ,cuMask)

        # cuTmp2 = cv2.cuda_GpuMat()
        # cuTmp2 = cv2.cuda.add(cuTmp1, cuRecColor)

        # cuTmp3 = cv2.cuda_GpuMat()
        # cuTmp3 = cv2.cuda.subtract(cuMask, ones)

        # cuTmp4 = cv2.cuda_GpuMat()
        # cuTmp4 = cv2.cuda.multiply(cuTmp2, cuTmp3)

        # cuResult = cv2.cuda_GpuMat()
        # cuResult = cv2.cuda.abs()

        #Making the Mask 3 channels, this may take time...
        recMask =  cuRecMask.download()
        recMask = np.dstack((recMask,recMask,recMask)).astype(dtype='uint8')
        ones.upload(np.full_like(uv_image, 1, dtype="uint8"))
        cuRecMask.upload(recMask)

        cuFinal = cv2.cuda_GpuMat();  cuFinal.upload(np.full_like(uv_image, 0, dtype="uint8"))
        cuFinal = cv2.cuda.abs( cv2.cuda.add( cv2.cuda.multiply(cuResult, cuRecMask) , cv2.cuda.multiply( cuRecColor , cv2.cuda.subtract(ones, cuRecMask) ) )  )


        # final_uv = fastAlphaBlend(rec_uv_tmp, rec_color_image, rec_mask )
        # def fastAlphaBlend(fg,bg,alpha):
        #     # MY VERSION
        #     a = alpha[:, :, np.newaxis]
        #     blended = cv2.convertScaleAbs(fg * a + bg * (1-a))
        #     return blended

        #  dst(I) = saturate  (  |src(I)∗alpha+beta|   ) 


        # mask = cuRecMask.download()
        # colorRec = cuRecColor.download()
        # uvRec = cuRecUv.download()
        # result = cuResult.download()
        final = cuFinal.download()

        # cv2.imshow('mask', mask)
        # cv2.imshow('color rec', colorRec)
        # cv2.imshow('uv rec', uvRec)
        # cv2.imshow('cu result', result)
        cv2.imshow('final', final)


    

        # return recLeft, recRight, recMask
        # return self.cutImage(recLeft, recRight)



        # mask = cuMask.download() 
        end = time.time_ns() 
        elapsed = (end - start) / 1000000
        print ("Elapsed: " + str(elapsed))

        # mask = cv2.convertScaleAbs(mask, alpha=np.average(mask)/255.0)
        # cv2.imshow('DEBUG', mask)
        # plt.imshow(mask)
        # plt.show()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
