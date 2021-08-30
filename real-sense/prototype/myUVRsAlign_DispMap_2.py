######################################################################################
##      Align RGB & UV cameras using the RS depth as reference                      ##
##      uses depth value of each pixel to estimate the disparity on the other image ##
######################################################################################
import pyrealsense2 as rs
from numpy.linalg import inv
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


resX = 1280
resY = 720
baseline_Uv = 17 #mm (not final value yet, confirm with calibration data)
fx_Uv = 4.2353589064127340e+02 #fx from camera instrinsics (since it is vertical stereo)  
Tdiv = 1000.0
Rtmul = 1.3
clipping_distance_in_meters = 1.15 #1 meter
alphaValue = 0.6  #alpha for the mask transparency
mapx1=None; mapx2=None; mapy1=None; mapy2=None
maxDist=1500.0    #in milimeters
scX = 1.0 
scY = 1.0

def erosion(src,eType,eSize):
    eSize = eSize
    eType = eType  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(eType, (2*eSize + 1, 2*eSize+1), (eSize, eSize))
    return cv.erode(src, element)

def dilatation(src,dType,dSize):
    dSize = dSize
    dType = dType  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(dType, (2*dSize + 1, 2*dSize+1), (dSize, dSize))
    return cv.dilate(src, element)

#This is a faster compositor
def fastAlphaBlend(fg,bg,alpha):
    '''
    Composit fg image onto bg image according to alpha. Alpha should be float [0.0, 1.0]
    uint8(fg * a + bg * (1-a))
    :param fg: Foreground image, should be same shape as bg
    :param bg: Background, should be same shape as fg
    :param alpha: Alpha mask image 32FC1 [0.0 1.0] image same widthxheight as foreground
    :return: 8U blended image
    '''
    #original version
    # a = (np.multiply(alpha, 1.0 / 255))[:,:,np.newaxis]
    # blended = cv.convertScaleAbs(fg * a + bg * (1-a))
    # MY VERSION
    a = alpha[:, :, np.newaxis]
    blended = cv.convertScaleAbs(fg * a + bg * (1-a))
    return blended

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

    mapx1, mapy1 = cv.initUndistortRectifyMap(M1, D1, R1, P1,(resX, resY), cv.CV_16SC2)
    mapx2, mapy2 = cv.initUndistortRectifyMap(M2, D2, R2, P2,(resX, resY), cv.CV_16SC2)

#Function to read the extrincics from the file
def loadCamFiles(fExt, fInt):
    #Load extrinsic matrix variables 
    fext = cv.FileStorage(fExt, cv.FILE_STORAGE_READ) 
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
    fint = cv.FileStorage(fInt, cv.FILE_STORAGE_READ) 
    M1 = fint.getNode("M1").mat()  #cameraMatrix[0]
    D1 = fint.getNode("D1").mat()  #distCoeffs[0]
    M2 = fint.getNode("M2").mat()  #cameraMatrix[1]
    D2 = fint.getNode("D2").mat()  #distCoeffs[2]

    # print("R")
    # print(R)
    # print("T")
    # print(T)

    ######  Rt*1.3 and T/26  WHYYYYYY ????? !!!! 
    #format extrinsics for OpenCV use
    Rt =  np.zeros((4,4), dtype=np.float32) #rotation matrix
    Rt[0][0] = R[0][0]*Rtmul; Rt[0][1] = R[0][1]*Rtmul;  Rt[0][2] = R[0][2]*Rtmul; Rt[0][3] = T[0]/Tdiv
    Rt[1][0] = R[1][0]*Rtmul; Rt[1][1] = R[1][1]*Rtmul;  Rt[1][2] = R[1][2]*Rtmul; Rt[1][3] = T[1]/Tdiv
    Rt[2][0] = R[2][0]*Rtmul; Rt[2][1] = R[2][1]*Rtmul;  Rt[2][2] = R[2][2]*Rtmul; Rt[2][3] = T[2]/Tdiv
    Rt[3][0] = 0; Rt[3][1] = 0;  Rt[3][2] = 0; Rt[3][3] = 1  
    # print ("Rt to UV")
    # print (Rt)

    # print ("RS cam intrinsics")
    # print (M1)
    # print ("UV cam intrinsics")
    # print (M2)
    fint.release()
    fext.release()
    return Rt, M2, D2

def loadCamFiles_2(fExt, fInt):
    #Load extrinsic matrix variables 
    fext = cv.FileStorage(fExt, cv.FILE_STORAGE_READ) 
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
    fint = cv.FileStorage(fInt, cv.FILE_STORAGE_READ) 
    M1 = fint.getNode("M1").mat()  #cameraMatrix[0]  #RGB
    D1 = fint.getNode("D1").mat()  #distCoeffs[0]    #RGB
    M2 = fint.getNode("M2").mat()  #cameraMatrix[1]  #UV
    D2 = fint.getNode("D2").mat()  #distCoeffs[2]    #UV
 
    fint.release()
    fext.release()

    # def undistort(M1, M2, D1, D2, R1, R2, P1, P2):
    undistort(M1, M2, D1, D2, R1, R2, P1, P2) 
    return Q, M2, D2



#rectifies the stereo camera pair
def remap(left, right, depth):  #left is rgb_RS, right is UV, depth is UV depth 
    #remap(mg, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR)
    recLeft = np.full_like(left, 0, dtype="uint8")
    recRight = np.full_like(right, 0, dtype="uint8")
    recDepth = np.full_like(depth, 0, dtype="float")

    recLeft = cv.remap(left, mapx1, mapy1, cv.INTER_LINEAR)  #rgb
    recRight = cv.remap(right, mapx2, mapy2, cv.INTER_LINEAR) #uv
    recDepth = cv.remap(depth, mapx2, mapy2, cv.INTER_LINEAR) #depth 
    #recRight = cv.warpPerspective(recRight, self.R1, (self.width, self.height))

    return recLeft, recRight, recDepth
    # return self.cutImage(recLeft, recRight)

#Disparity from depthMap (depthMap should be given in meters)
baseLineY = 27.534100646130540  #taken from extrinsics T vector [ -1.1517719701854450e+01, 2.3134096167196148e+01,-1.0639161017443953e+01 ]
fy = 695.19449163456534   # taken from extrinsics  6.9519449163456534e+02  8.0002217022260618e+02,
# fy = fy / 3.0
def getDisparityMap(depthMap):
    base = baseLineY * fy 

    # # old valid values 
    # # good base 13133.0205078125 px/mm 
    # # good T  27.93714518  mm
    # # good F  470.0917156422387   px ??
    # # real F  951.27001953125
    dispMap = np.zeros((resY,resX), dtype=np.float)
    # np.multiply(dispMap, 4.0)

    for y in range(0, resY, 1):
        for x in range(0, resX, 1):
            if (depthMap[y][x] > 0.02):
                Z = depthMap[y][x] * 1000.0
                if (Z < 1500.0):
                    dispMap[y][x] = base / Z  #in pixels (in theory ) disp in Y
    return dispMap

    #SLOW AS HELL VERSION !!!
    # dispMapX = np.full((resY, resX), baseLineX * rgb_uv_f)
    # dispMapY = np.full((resY, resX), baseLineY * rgb_uv_f)
    # depthMap = np.multiply(depthMap , 1000.0)

    # dispMapX = np.divide(dispMapX, depthMap)
    # dispMapY = np.divide(dispMapY, depthMap)
    # dispMapX[np.isinf(dispMapX)] = 0.0
    # dispMapY[np.isinf(dispMapY)] = 0.0

    # return np.stack((dispMapX, dispMapY))  #the extra () are important  

def getUVTexture(back, front ,mask, dispMap):
    for x in range(0, resX, 1):
        for y in range(0, resY, 1):
            if (mask[y][x] > 0):
                if ( 0 <=  (y + int(round(dispMap[y][x]))* scY)  < resY):
                    front[y][x][0] = front[y + int(round(dispMap[y][x]) * scY )] [x][0]  
                    front[y][x][1] = front[y + int(round(dispMap[y][x]) * scY )] [x][1]
                    front[y][x][2] = front[y + int(round(dispMap[y][x]) * scY )] [x][2]  

    # final = fastAlphaBlend(uv_image, color_image, mask)
    return front

def raw_depth_to_meters(raw_depth):
    dist = np.zeros((resY,resX), dtype=np.float)
    for x in range(0, resX, 1):
        for y in range(0, resY, 1):
            if (raw_depth[y][x] < 2047):
                dist[y][x] = 1.0 / (raw_depth[y][x] * -0.0030711016 + 3.3309495161)
    return dist


cap = cv.VideoCapture(3)  #open chinese UV camera 3

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, resX, resY, rs.format.z16, 30)
config.enable_stream(rs.stream.color, resX, resY, rs.format.bgr8, 30)

# Start streaming
cfg = pipeline.start(config)

# RS colorizer object
colorizer = rs.colorizer()

# Getting the depth sensor's depth scale (see rs-align example for explanation)
# depth_sensor = cfg.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# Gets RealScene intrinsic parameters
i_profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = i_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
c_profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
c_intr = c_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away



#Build RS depth intrinsics in OpenCV matrix format
# K = [fx 0 cx; 
#      0 fy cy; 
#      0  0  1]
depth_cam_matrix = np.zeros((3,3), dtype=np.float)
depth_cam_matrix[0][0] =  intr.fx #fx
depth_cam_matrix[0][2] =  intr.ppx #cx
depth_cam_matrix[1][1] =  intr.fy #fy
depth_cam_matrix[1][2] =  intr.ppy #cy
depth_cam_matrix[2][2] = 1
# print ("RS infrared intrinsics")
# print (depth_cam_matrix)

# scaled_depth = depth_cam_matrix.copy()
# scaled_depth[0][0] = intr.fx / 1.364740861736523
# scaled_depth[1][1] = intr.fy / 1.364740861736523
# print (scaled_depth)

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

# RS extrinsics to OpenCV translation vector and rotation matrix 
# Get the extrinsics between the cameras from the calibration YML file 
Rt_to_Uv, uv_cam_matrix, uv_cam_distort = loadCamFiles('./rs_params/extrinsics.yml', './rs_params/intrinsics.yml')
rgb_uv_Q, M_2, D_2 = loadCamFiles_2('./rgb_uv_params/extrinsics.yml', './rgb_uv_params/intrinsics.yml')


# Gets RealSense camera
extrinsics = i_profile.get_extrinsics_to(c_profile)

Rt_to_Rgb = np.zeros((4,4), dtype=np.float32) #rotation matrix
Rt_to_Rgb[0][0] = extrinsics.rotation[0]; Rt_to_Rgb[0][1] = extrinsics.rotation[1];  Rt_to_Rgb[0][2] = extrinsics.rotation[2]; Rt_to_Rgb[0][3] = extrinsics.translation[0] 
Rt_to_Rgb[1][0] = extrinsics.rotation[3]; Rt_to_Rgb[1][1] = extrinsics.rotation[4];  Rt_to_Rgb[1][2] = extrinsics.rotation[5]; Rt_to_Rgb[1][3] = extrinsics.translation[1] 
Rt_to_Rgb[2][0] = extrinsics.rotation[6]; Rt_to_Rgb[2][1] = extrinsics.rotation[7];  Rt_to_Rgb[2][2] = extrinsics.rotation[8]; Rt_to_Rgb[2][3] = extrinsics.translation[2]
Rt_to_Rgb[3][0] = 0; Rt_to_Rgb[3][1] = 0;  Rt_to_Rgb[3][2] = 0; Rt_to_Rgb[3][3] = 1  
# print ("Rt to RGB")
# print (Rt_to_Rgb)

# Check if the UV camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

#change webcam resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH,resX)  
cap.set(cv.CAP_PROP_FRAME_HEIGHT,resY)  

#parameters needed to translate the UV depthmap after scaling
tX = uv_cam_matrix[0][2] - uv_cam_matrix[0][2] * 1.3
tY = uv_cam_matrix[1][2] - uv_cam_matrix[1][2] * 1.3
T = np.float32([[1, 0, tX], [0, 1, tY]])


dev = cfg.get_device()
depth_sensor = dev.first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, True)  # INFRARED PROJECTOR TURN ON

# if (depth_sensor.supports(rs.option.emitter_enabled)):
#     depth_sensor.set_option(rs.option.emitter_enabled, False)  #INFRARED PROJECTOR TURN OFF

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor.set_option(rs.option.depth_units, 0.01)
depth_scale = depth_sensor.get_depth_scale()
clipping_distance = clipping_distance_in_meters / depth_scale
print("Depth Scale is: " , depth_scale)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ret, uv_image = cap.read()  #uv camera
        if not depth_frame or ret == False or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # MAIN REGISTRATION FUNCTION    
        #  uv_rgb = K_rgb * [R | t] * z * inv(K_ir) * uv_ir 
        # unregisteredCameraMatrix	the camera matrix of the depth camera  (depth_cam_matrix)
        # registeredCameraMatrix	the camera matrix of the external camera (uv_cam_matrix)
        # registeredDistCoeffs	    the distortion coefficients of the external camera  (infra_cam_dist)
        # Rt	                    the rigid body transform between the cameras. Transforms points from depth camera frame to external camera frame. (Rt)
        # unregisteredDepth	        the input depth data  (depth_image)
        # outputImagePlaneSize      the image plane dimensions of the external camera (width, height)
        depth_image2 = depth_image.astype(dtype=np.float)*depth_scale
        uv_regDepth = cv.rgbd.registerDepth( depth_cam_matrix, M_2, None, Rt_to_Uv, depth_image2, (resX, resY), depthDilation=False)
        # uv_regDepth = cv.rgbd.registerDepth( depth_cam_matrix, uv_cam_matrix, uv_cam_distort, Rt_to_Uv, depth_image2, (resX, resY), depthDilation=False)
        # color_regDepth = cv.rgbd.registerDepth( depth_cam_matrix, color_cam_matrix, color_distort, Rt_to_Rgb, depth_image, (resX, resY), depthDilation=False)
        # uv_regDepth = uv_regDepth.astype(dtype=np.float)*depth_scale #from UINT16 to FLOAT (in meters)
        # uv_regDepth = cv2.cvtColor(uv_regDepth, cv2.CV_16UC1)
        
        distance = depth_image2
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(distance)
        f.add_subplot(1,2, 2)
        distance2 = uv_regDepth 
        plt.imshow(distance2)
        plt.show(block=True)

        color_image, uv_image, uv_regDepth = remap(color_image, uv_image, distance2)  #Rectify the images
        
        #rescale the aligned depth map and the UV camera image 
        # i DO NOT REALLY NOW WHAT IS THIS BLOCK ABOUT, BUT IN THE LAST VERSION IT WAS USEFULL. 
        # uv_image = cv.resize(uv_image, None, fx=1.3,fy=1.3, interpolation=cv.INTER_CUBIC)  #scale up UV depth map
        # uv_image = cv.warpAffine(uv_image, T, (resX,resY))
        # uv_resized = cv.resize(uv_regDepth, None, fx=1.3,fy=1.3, interpolation=cv.INTER_CUBIC)  #scale up UV depth map
        # uv_resized = cv.warpAffine(uv_resized, T, (resX,resY))
        uv_resized = uv_regDepth  # NOTE THIS !!!! WHY ???
        

        #EROTE AND DILATATE THE DEPTH MASK 
        mask = np.where((uv_resized > clipping_distance_in_meters) | (uv_resized <= 0.0), 0.0, alphaValue)  
        mask = erosion(mask,cv.MORPH_RECT, 10)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = dilatation(mask,cv.MORPH_RECT, 15)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = cv.GaussianBlur( mask,(15 , 15), 0, 0 )

        #get the disparity map from the depth map using the stereo formula with different values 
        uv_DispMap = getDisparityMap(uv_resized)

        
        # print (np.average( uv_DispMap[0] ))
        # print (np.amax( uv_DispMap[0] ))
        # xcx = cv.convertScaleAbs(uv_DispMap[0], alpha=255.0/np.amax( uv_DispMap[0] ))
        # xcxh = cv.applyColorMap( xcx, cv.COLORMAP_HOT )
        # xcx = cv.cvtColor(xcx, cv.COLOR_GRAY2BGR)
        
        
        #get the UV pixels as indicated by the disparity map
        uv_image = getUVTexture(color_image, uv_image, mask, uv_DispMap) 

        #  FEATURE EXTRACTION BY CANNY    FOR DEBUG ONLY instead of getUVTexture
        # Apply Canny to detect the borders
        # uvGray  = cv.cvtColor(uv_image, cv.COLOR_BGR2GRAY)
        # rgbGray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

        # # bilineal cauchy for border extraction
        # uvGray = cv.bilateralFilter(uvGray, 7, 20, 20)   
        # uvGray = cv.Canny(uvGray, 60, 150)    
        # rgbGray = cv.bilateralFilter(rgbGray, 7, 20, 20)
        # rgbGray = cv.Canny(rgbGray, 60, 150)    

        # masko =  cv.convertScaleAbs(mask, alpha=alphaValue)  #mask to 8 bit
        # uvGray = cv.bitwise_and(uvGray, uvGray, mask=masko)
        # rgbGray = cv.bitwise_and(rgbGray, rgbGray, mask=masko)

        # #feature extraction by Shi-Tomasi
        # corners1 = cv.goodFeaturesToTrack(uvGray, maxCorners=10000, qualityLevel=0.01, minDistance=5, mask=masko ) #original qualityLevel=0.01 minDistance=10
        # if (corners1 is not None):   
        #     corners1 = np.int0(corners1)
        #     corners1 = np.squeeze(corners1)

        #     corners2 = []
        #     for i in range(0, len(corners1), 1):
        #         corners2.append( (corners1[i][0] + int(uv_DispMap [corners1[i][1]] [corners1[i][0]] [0]), corners1[i][1] - int(uv_DispMap [corners1[i][1]] [corners1[i][0]] [1] )))
                
        #     # DEBUG draw the detected features (corners)
        #     for i in range(0, len (corners1), 1):
        #         uv_image = cv.circle(uv_image,(corners1[i][0], corners1[i][1]) ,2,(255,0,255),-1)
        #     for i in range(0, len(corners2), 1):
        #         color_image = cv.circle(color_image,(corners2[i][0], corners2[i][1]) ,2,(255,255,0),-1)
                


        #join the uv and the rgb images
        final = fastAlphaBlend(uv_image, color_image, mask)

        # uv_DispMap =  cv.convertScaleAbs(uv_DispMap, alpha=255/(np.average(uv_DispMap)))
        # cv.imshow('disparity map', uv_DispMap)
        cv.imshow('rgb', color_image)
        cv.imshow('uv', uv_image)
        cv.imshow('Aligned UV/RGB', final)

        # cv.imshow('mask debug', mask)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == ord('a'):
            scX = scX - 0.05
            print ("scX:"+ str(scX))
        elif key == ord('d'):
            scX = scX + 0.05
            print ("scX:"+ str(scX))
        elif key == ord('w'):
            fy = fy + 10.0
            print ("fy:"+ str(fy))
        elif key == ord('s'):
            fy = fy - 10.0
            print ("fy:"+ str(fy))

finally:
    # Stop streaming
    pipeline.stop()



