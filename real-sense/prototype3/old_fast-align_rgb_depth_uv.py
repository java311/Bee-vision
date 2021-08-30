import pyrealsense2 as rs
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt  #matplotlib for DEBUG only

#PROGRAM CONSTANTS 
mapx1=None; mapx2=None; mapy1=None; mapy2=None
alphaValue = 0.6  #alpha for the mask transparency
resX = 1280
resY = 720
# baseLineX = 0.0  #taken from extrinsics (loadCamFiles)
# f = 0.0          #taken from extrinsics (loadCamFiles)


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
def undistort(M1, M2, D1, D2, R1, R2, P1, P2):
    # PROBAR ESTA OPCION !!! OPCIONAL 1!!
    # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    #Mat rmap[2][2];
    global mapx1; global mapx2; global mapy1; global mapy2;
    mapx1 = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy1 = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapx2 = np.ndarray(shape=(resY, resX, 1), dtype='float32')
    mapy2 = np.ndarray(shape=(resY, resX, 1), dtype='float32')

    mapx1, mapy1 = cv2.initUndistortRectifyMap(M1, D1, R1, P1,(resX, resY), cv2.CV_16SC2)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2,(resX, resY), cv2.CV_16SC2)

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
    baseLine = T[1]

    #Load intrinsic matrix variables  (only for UV chinese camera)
    # M1 is the Real Sense !!! 
    fint = cv2.FileStorage(fInt, cv2.FILE_STORAGE_READ) 
    M1 = fint.getNode("M1").mat()  #cameraMatrix[0]  #RGB
    D1 = fint.getNode("D1").mat()  #distCoeffs[0]    #RGB
    M2 = fint.getNode("M2").mat()  #cameraMatrix[1]  #UV
    D2 = fint.getNode("D2").mat()  #distCoeffs[2]    #UV
 
    fint.release()
    fext.release()

    # take the distortion arguments from the extrinsic parameters
    # global baseLineX; global f
    # baseLineX = T[1]  #taken from extrinsics T vector 
    # f = Q[2][3]  # taken from extrinsics  Q matrix
    # print (baseLineX)
    # print (f)

    # calculate the distortion matrices
    undistort(M1, M2, D1, D2, R1, R2, P1, P2) 

#rectifies the stereo camera pair
def remap(left, right, mask ):  #left is rgb_RS, right is UV, depth is UV depth 
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

baseLineX = 32.324839335022240   #T[0]
fx = 840.62992309768606          #M1 fx
base = fx * baseLineX
def getDisparityMaps(depthMap):
    #first remap the depthMap
    recDepthMap = np.full_like(depthMap, 0, dtype="float32")
    recDepthMap = cv2.remap(depthMap, mapx1, mapy1, cv2.INTER_LINEAR) 

    dispMapX = 1.0/(recDepthMap.astype(np.float32))
    dispMapX = dispMapX * base
    
    grid = np.indices((resY, resX))
    dispMapX = grid[1].astype(np.float32) - dispMapX - 18 
    dispMapY = grid[0].astype(np.float32) 

    # dispMap = 1/(recDepthMap.astype(np.float32))
    # dispMap = dispMap * base


   
    # grid = np.indices((resY, resX))
    # dispMap = grid[0].astype(np.float32) + dispMap
    # dispx = grid[1].astype(np.float32)
    # dispx = mapx2
    return dispMapX, dispMapY 


def fastUVReMap(front, dispMapX, dispMapY):
    # disp_x = np.zeros((resY,resX), dtype=np.float32)
    
    result = np.zeros((resY,resX), dtype="uint8")

    # disp_y = np.zeros((resY,resX), dtype=np.float32)
    # plt.imshow(dispMap)
    # plt.pause(0.05)
    # plt.show()

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
cap = cv2.VideoCapture(1)  #open chinese UV camera 3

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
if (cap.isOpened() == False):
    print("Error opening video stream or file")

#change webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resX)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,resY)  

loadCamFiles('./cam_params/rgb_uv_extrinsics.yml', './cam_params/rgb_uv_intrinsics.yml')

xerr = 0
yerr = 0

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
        ret, uv_image = cap.read()  #uv camera

        # Validate that both frames are valid
        if not aligned_depth_frame or ret == False or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #FAKE uv_color for DEBUG !!!
        # channels = cv2.split(uv_image)
        # uv_image = np.dstack((channels[0] * 0,channels[2],channels[1]))
        
        # Remove background - Set pixels further than clipping_distance to grey
        # grey_color = 153
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        mask = np.where((depth_image > clipping_distance) | (depth_image <= 0.0), 0.0, alphaValue) 
        mask = erosion(mask,cv2.MORPH_RECT, 10)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = dilatation(mask,cv2.MORPH_RECT, 15)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = cv2.GaussianBlur( mask,(15 , 15), 0, 0 )

        disp_x, disp_y = getDisparityMaps(depth_image)  #Transforms the depth map to disparity map 

        rec_color_image, rec_uv_image, rec_mask = remap(color_image, uv_image, mask)  #Rectify UV, RGB and MASK
        # T = np.float32([[1, 0, xerr], [0, 1, yerr]]) # with camera calibration parameters
        # rec_uv_image = cv2.warpAffine(rec_uv_image, T, (resX,resY))
        rec_uv_tmp = fastUVReMap(rec_uv_image, disp_x, disp_y)

        # rec_uv_fcolor =  fakeColor(rec_uv_tmp, rec_color_image)  #THIS "HAS A FUCKING BUG"
        cv2.imshow('debugo', rec_uv_tmp)
        cv2.imshow('debugo2', rec_uv_image)
        # mask = np.full((resY, resX), alphaValue) #DEBUG
        final = fastAlphaBlend(rec_uv_tmp, rec_color_image, rec_mask )

        #######################################DEBUG BY CAUCHY###############################################

        # #  FEATURE EXTRACTION BY CANNY 
        # # Apply Canny to detect the borders
        # uvGray  = cv2.cvtColor(rec_uv_image, cv2.COLOR_BGR2GRAY)
        # rgbGray = cv2.cvtColor(rec_color_image, cv2.COLOR_BGR2GRAY)

        # # bilineal cauchy for border extraction
        # uvGray = cv2.bilateralFilter(uvGray, 7, 20, 20)   
        # uvGray = cv2.Canny(uvGray, 60, 150)    
        # rgbGray = cv2.bilateralFilter(rgbGray, 7, 20, 20)
        # rgbGray = cv2.Canny(rgbGray, 60, 150)    

        # # masko =  cv2.convertScaleAbs(mask, alpha=alphaValue)  #mask to 8 bit
        # # uvGray = cv2.bitwise_and(uvGray, uvGray, mask=masko)
        # # rgbGray = cv2.bitwise_and(rgbGray, rgbGray, mask=masko)

        # # mask = np.full((resY, resX), alphaValue) #DEBUG
        # # depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
        # final = fastAlphaBlend(rec_disp_image, rec_color_image, mask)

        # #feature extraction by Shi-Tomasi
        # corners1 = cv2.goodFeaturesToTrack(rgbGray, maxCorners=10000, qualityLevel=0.01, minDistance=5 ) #original qualityLevel=0.01 minDistance=10
        # if (corners1 is not None):
        #     corners1 = np.int0(corners1)
        #     corners1 = np.squeeze(corners1)

        #     corners2 = []
        #     for i in range(0, len (corners1), 1):
        #         if rec_disp_image [corners1[i][1]] [corners1[i][0]] > 0:  #filter negative and zero disparities 
        #             corners2.append (  ( int(corners1[i][0]) , int(corners1[i][1] + rec_disp_image [corners1[i][1]] [corners1[i][0]] ) ) )

        #     # DEBUG draw the detected features (corners)
        #     for i in range(0, len (corners1), 1):
        #         final = cv2.circle(final,(corners1[i][0], corners1[i][1]) ,2,(0,0,255),-1)  # RGB red (BGR)
        #     for i in range(0, len(corners2), 1):
        #         final = cv2.circle(final,(corners2[i][0], corners2[i][1]) ,2,(0,255,0),-1)  #UV green

        #######################################DEBUG BY CAUCHY###############################################

        # rec_uv_image_2 = rectifyUV(rec_color_image, rec_uv_image, mask, rec_disp_image
        # Render images
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(rec_disp_image, alpha=0.03), cv2.COLORMAP_JET)
        
        end = time.time_ns() 
        elapsed = (end - start) / 1000000
        # print (elapsed)

        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', final)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('a'):
            fx = fx + 10.0
            print ("fx:" + str(fx))
        elif key & 0xFF == ord('d'):
            fx = fx - 10.0
            print ("fx:" + str(fx))
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
