import pyrealsense2 as rs
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt  #matplotlib for DEBUG only

#PROGRAM CONSTANTS 
mapx1_uv=None; mapx2_uv=None; mapy1_uv=None; mapy2_uv=None
mapx1_infra=None; mapx2_infra=None; mapy1_infra=None; mapy2_infra=None
alphaValue = 0.6  #alpha for the mask transparency
resX = 1280
resY = 720
err = 0.0 #DEBUG


#This is a faster compositor
def fastAlphaBlend(fg,bg,alpha):
    # MY VERSION
    a = alpha[:, :, np.newaxis]
    blended = cv2.convertScaleAbs(fg * a + bg * (1-a))
    return blended

def erosion(src,eType,eSize):
    eSize = eSize
    eType = eType  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE
    element = cv2.getStructuringElement(eType, (2*eSize + 1, 2*eSize+1), (eSize, eSize))
    return cv2.erode(src, element)

def dilatation(src,dType,dSize):
    dSize = dSize
    dType = dType  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dType, (2*dSize + 1, 2*dSize+1), (dSize, dSize))
    return cv2.dilate(src, element)

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

    mapx1_uv, mapy1_uv = cv2.initUndistortRectifyMap(M1, D1, R1, P1,(resX, resY), cv2.CV_16SC2)
    mapx2_uv, mapy2_uv = cv2.initUndistortRectifyMap(M2, D2, R2, P2,(resX, resY), cv2.CV_16SC2)


#calculates the rectify matrices from the camera parameters
def undistortINFRA(M1, M2, D1, D2, R1, R2, P1, P2):
    # PROBAR ESTA OPCION !!! OPCIONAL 1!!
    # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    #Mat rmap[2][2];
    global mapx1_infra; global mapx2_infra; global mapy1_infra; global mapy2_infra;
    mapx1_infra = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy1_infra = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapx2_infra = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy2_infra = np.ndarray(shape=(resY, resX, 1), dtype='float32')

    mapx1_infra, mapy1_infra = cv2.initUndistortRectifyMap(M1, D1, R1, P1,(resX, resY), cv2.CV_16SC2)
    mapx2_infra, mapy2_infra = cv2.initUndistortRectifyMap(M2, D2, R2, P2,(resX, resY), cv2.CV_16SC2)

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
    # M1 is the Real Sense !!! 
    fint = cv2.FileStorage(fInt, cv2.FILE_STORAGE_READ) 
    M1 = fint.getNode("M1").mat()  #cameraMatrix[0]  #RGB
    D1 = fint.getNode("D1").mat()  #distCoeffs[0]    #RGB
    M2 = fint.getNode("M2").mat()  #cameraMatrix[1]  #UV
    D2 = fint.getNode("D2").mat()  #distCoeffs[2]    #UV
 
    fint.release()
    fext.release()

    # return baseline and focal length
    # if Mindex[0] == 1:
    #     return abs(float(T[Tindex])), abs(M1[Mindex[1]][Mindex[2]]), [M1, M2, D1, D2, R1, R2, P1, P2, T]
    # elif Mindex[0] == 2:
    #     return abs(float(T[Tindex])), abs(M2[Mindex[1]][Mindex[2]]), [M1, M2, D1, D2, R1, R2, P1, P2, T]

    return [M1, M2, D1, D2, R1, R2, P1, P2, T]

    # baseLineX = 32.324839335022240   #T[0]
    # fx = 840.62992309768606          #M1 fx
    # base = fx * baseLineX

#rectifies the stereo camera pair
def remap(left, right, mask, mapx1, mapy1, mapx2, mapy2):  #left is rgb_RS, right is UV, depth is UV depth 
    #remap(mg, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR)
    recLeft = np.full_like(left, 0, dtype="uint8")
    recRight = np.full_like(right, 0, dtype="uint8")
    recMask = np.full_like(mask, 0, dtype="float")

    recLeft  = cv2.remap(left,  mapx1, mapy1, cv2.INTER_LINEAR)  #rgb
    recRight = cv2.remap(right, mapx2, mapy2, cv2.INTER_LINEAR) #uv
    recMask = cv2.remap(mask, mapx1, mapy1, cv2.INTER_LINEAR) #mask aligned to rgb
    #recRight = cv2.warpPerspective(recRight, self.R1, (self.width, self.height))

    return recLeft, recRight, recMask
    # return self.cutImage(recLeft, recRight)

# baseLineX = 32.324839335022240   #T[0]
# fx = 840.62992309768606          #M1 fx
# base = fx * baseLineX
# Calculates the disparity for images that are rectified vertically (Y)
# So it returns the rectified map on X
def getHorizontalDisparity(depthMap, base, mapx, mapy, positive, err):
    #first remap the depthMap
    recDepthMap = np.full_like(depthMap, 0, dtype="float32")
    recDepthMap = cv2.remap(depthMap, mapx, mapy, cv2.INTER_LINEAR) 

    dispMapX = 1.0/(recDepthMap.astype(np.float32))
    dispMapX = dispMapX * base
    
    grid = np.indices((resY, resX))
    if positive:
        dispMapX = grid[1].astype(np.float32) + dispMapX 
    else:
        dispMapX = grid[1].astype(np.float32) - dispMapX 
    dispMapY = grid[0].astype(np.float32) 

    return dispMapX, dispMapY 


# Calculates the disparity for images that are horizontally rectified (X), 
# So it returns the rectify map on Y
def getVerticalDisparity(depthMap, base, mapx, mapy, positive, err):
    #first remap the depthMap
    recDepthMap = np.full_like(depthMap, 0, dtype="float32")
    recDepthMap = cv2.remap(depthMap, mapx, mapy, cv2.INTER_LINEAR) 

    dispMapY = 1.0/(recDepthMap.astype(np.float32))
    dispMapY = dispMapY * base
    
    grid = np.indices((resY, resX))
    if (positive):
        dispMapY = grid[0].astype(np.float32) + dispMapY + err
    else:
        dispMapY = grid[0].astype(np.float32) - dispMapY - err
    dispMapX = grid[1].astype(np.float32) 

    return dispMapX, dispMapY 




def fastReMap(front, dispMapX, dispMapY):    
    result = np.zeros((resY,resX), dtype="uint8")
    result = cv2.remap(front, dispMapX, dispMapY, cv2.INTER_LINEAR)

    return result

def fakeColor(uv_image,rgb_image):
    #Bee vision 1. RGB ((0,100,0)%,(0,0,100)%,(0,0,0)%)
    gb = np.full_like(rgb_image, 0, dtype="uint8")
    channels = cv2.split(rgb_image) #BGR split
    gb = np.dstack((channels[0] * 0,channels[2],channels[1]))
    # cv2.imshow('gb', gb)

    #Bee vision 2. UV (0%,0%, (40%,40%,20%) ) 
    uv_blue = np.full_like(uv_image, 0, dtype="uint8")
    channels = cv2.split(uv_image) #BGR split
    tmp1 = cv2.addWeighted(channels[0],0.2,channels[1],0.4,0) 
    tmp2 = cv2.addWeighted(tmp1,0.6,channels[2],0.4,0)
    uv_blue = np.dstack((tmp2,channels[1]*0,channels[2]*0,))
    # cv2.imshow("uv_blue", uv_blue)

    return cv2.addWeighted(uv_blue,0.6,gb,0.4,0)

    
#interactive matlab plot
plt.ion() 

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
# base_rgb_uv = abs(float(Mats[8][1])) * abs(float(Mats[0][1][1]))


mats = loadCamFiles('./cam_params/rgb_infra_extrinsics.yml', './cam_params/rgb_infra_intrinsics.yml')  # cause rectifed on Y
undistortINFRA(mats[0], mats[1], mats[2], mats[3], mats[4], mats[5], mats[6], mats[7])  # calculate the distortion matrices
base_rgb_infra = abs(float(mats[8][1])) * abs(float(mats[0][1][1]))                  # T[0]  M1[fy] cause rectified on X (vertical cameras)
# print (base_rgb_infra)  

# xerr = 0
# yerr = 0
# Streaming loop
try:
    while True:
        start = time.time_ns() 

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        ret, uv_image = cap2.read()  #uv camera
        ret2, infra_image = cap.read()  #infra camera

        # Validate that both frames are valid
        if not aligned_depth_frame or ret == False or ret2 == False or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imwrite('depth_image.png', depth_image)
        cv2.imwrite('color_image.png', color_image)
        cv2.imwrite('uv_image.png', uv_image)

        #FAKE uv_color for DEBUG !!!
        # channels = cv2.split(uv_image)
        # uv_image = np.dstack((channels[0] * 0,channels[2],channels[1]))
        uv_image = cv2.cvtColor(uv_image, cv2.COLOR_BGR2GRAY)
        uv_image = cv2.applyColorMap(uv_image, cv2.COLORMAP_HOT)

        # Remove background - Set pixels further than clipping_distance to grey
        mask = np.where((depth_image > clipping_distance) | (depth_image <= 0.0), 0.0, alphaValue) 
        mask = erosion(mask,cv2.MORPH_RECT, 10)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = dilatation(mask,cv2.MORPH_RECT, 15)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = cv2.GaussianBlur( mask,(15 , 15), 0, 0 )
        # mask = np.full((resY, resX), alphaValue) #DEBUG

        ##### UV image rectification ##############
        disp_x, disp_y = getHorizontalDisparity(depth_image, base_rgb_uv, mapx1_uv, mapy1_uv, True, err)  #Transforms the depth map to disparity map 
        rec_color_image, rec_uv_image, rec_mask = remap(color_image, uv_image, mask, mapx1_uv, mapy1_uv, mapx2_uv, mapy2_uv)  #Rectify UV, RGB and MASK
        rec_uv_tmp = fastReMap(rec_uv_image, disp_x, disp_y)
        # rec_uv_fcolor =  fakeColor(rec_uv_tmp, rec_color_image)  #BUG
        cv2.imshow('debugo1', rec_uv_image)
        cv2.imshow('debugo2', rec_uv_tmp)
        final_uv = fastAlphaBlend(rec_uv_tmp, rec_color_image, rec_mask )
        ##### UV image rectification ##############


        # ##### INFRARED image rectification ##############
        # disp_x, disp_y = getVerticalDisparity(depth_image, base_rgb_infra, mapx1_infra, mapy1_infra, True, 9)  #Transforms the depth map to disparity map 
        # rec_color_image, rec_infra_image, rec_mask = remap(color_image, infra_image, mask, mapx1_infra, mapy1_infra, mapx2_infra, mapy2_infra)  #Rectify INFRA, RGB and MASK
        # rec_infra_tmp = fastReMap(rec_infra_image, disp_x, disp_y)
        # rec_infra_fcolor =  fakeColor(rec_infra_tmp, rec_color_image)  #BUG
        # # cv2.imshow('debugo', rec_infra_tmp)
        # # cv2.imshow('debugo2', rec_infra_image)
        # final_infra = fastAlphaBlend(rec_infra_fcolor, rec_color_image, rec_mask )
        # ##### INFRARED image rectification ##############

        end = time.time_ns() 
        elapsed = (end - start) / 1000000
        print ("Elapsed: " + str(elapsed))

        # cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Ultraviolet Rectified', final_uv)
        # cv2.imshow('Infra Rectified', final_infra)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('a'):
            err = err + 1.0
            print ("err:" + str(err))
        elif key & 0xFF == ord('d'):
            err = err - 1.0
            print ("err:" + str(err))
        if key & 0xFF == ord('w'):
            yerr = yerr + 1.0
            print ("yerr:" + str(yerr))
        elif key & 0xFF == ord('s'):
            yerr = yerr - 1.0
            print ("yerr:" + str(yerr))
        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
