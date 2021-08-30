##############################################################
##      Align RGB & UV cameras using the RS depth as reference  ##
##############################################################

import pyrealsense2 as rs
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import cv2 as cv

resX = 1280
resY = 720
baseline_Uv = 17 #mm (not final value yet, confirm with calibration data)
fx_Uv = 4.2353589064127340e+02 #fx from camera instrinsics (since it is vertical stereo)  
Tdiv = 1000.0
Rtmul = 1.3

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
    

def blendAlpha(foreground, background, alpha):
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv.add(foreground, background)

    return outImage/255

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

    #Load intrinsic matrix variables  (only for UV chinese camera)
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

    #get valid rectangle of the calibration for (RS vs UV cameras)
    # n = fext.getNode("validRoi1")
    # m = fext.getNode("validRoi2")
    # vroi1 = []
    # vroi2 = []
    # for i in range(n.size()):
    #     vroi1.append(int(n.at(i).real()))  #x,y,w,h
    #     vroi2.append(int(m.at(i).real()))  #x,y,w,h

    #get the largest rectangle cut
    # vroi1 = fext.getNode("validRoi1").mat()
    # vroi2 = fext.getNode("validRoi2").mat()
    # Vroi = [] 
    # Vroi.append(min(vroi1[0],vroi2[0])) #x
    # Vroi.append(min(vroi1[1],vroi2[1])) #y
    # Vroi.append(max(vroi1[2],vroi2[2])) #w
    # Vroi.append(max(vroi1[3],vroi2[3])) #h

    fint.release()
    fext.release()
    return Rt, M2, D2


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
depth_sensor = cfg.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# Gets RealScene intrinsic parameters
i_profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = i_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
c_profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
c_intr = c_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

dev = cfg.get_device()
depth_sensor = dev.first_depth_sensor()
# depth_sensor.set_option(rs.option.emitter_enabled, True)  # INFRARED PROJECTOR TURN ON

if (depth_sensor.supports(rs.option.emitter_enabled)):
    depth_sensor.set_option(rs.option.emitter_enabled, False)  #INFRARED PROJECTOR TURN OFF

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.0 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

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
print ("RS infrared intrinsics")
print (depth_cam_matrix)

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

# Gets RealSense camera
extrinsics = i_profile.get_extrinsics_to(c_profile)

Rt_to_Rgb = np.zeros((4,4), dtype=np.float32) #rotation matrix
Rt_to_Rgb[0][0] = extrinsics.rotation[0]; Rt_to_Rgb[0][1] = extrinsics.rotation[1];  Rt_to_Rgb[0][2] = extrinsics.rotation[2]; Rt_to_Rgb[0][3] = extrinsics.translation[0] 
Rt_to_Rgb[1][0] = extrinsics.rotation[3]; Rt_to_Rgb[1][1] = extrinsics.rotation[4];  Rt_to_Rgb[1][2] = extrinsics.rotation[5]; Rt_to_Rgb[1][3] = extrinsics.translation[1] 
Rt_to_Rgb[2][0] = extrinsics.rotation[6]; Rt_to_Rgb[2][1] = extrinsics.rotation[7];  Rt_to_Rgb[2][2] = extrinsics.rotation[8]; Rt_to_Rgb[2][3] = extrinsics.translation[2]
Rt_to_Rgb[3][0] = 0; Rt_to_Rgb[3][1] = 0;  Rt_to_Rgb[3][2] = 0; Rt_to_Rgb[3][3] = 1  
print ("Rt to RGB")
print (Rt_to_Rgb)

# Check if the UV camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

#change webcam resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH,resX)  
cap.set(cv.CAP_PROP_FRAME_HEIGHT,resY)  

tX = uv_cam_matrix[0][2] - uv_cam_matrix[0][2] * 1.3
tY = (uv_cam_matrix[1][2] - uv_cam_matrix[1][2] * 1.3) 
T = np.float32([[1, 0, tX], [0, 1, tY]]) 

#Gstreamer object
# video = cv.VideoWriter('appsrc ! videoconvert ! '
#                       'x264enc noise-reduction=10000 speed-preset=ultrafast tune=zerolatency ! '
#                       'rtph264pay config-interval=1 pt=96 !'
#                       'tcpserversink host=192.168.1.27 port=5000 sync=false',
#                       0, 25.0, (640, 480))

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ret, infra_image = cap.read()  #uv camera
        if not depth_frame or ret == False or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #DEBUG PAINT THE CORNER OF THE DEPTH IMAGE IN WHITE 
        for i in range(0,depth_image.shape[1],1):
            depth_image[0][i] = 32767
        for i in range(0,depth_image.shape[0],1):
            depth_image[i][0] = 32767
        
        # MAIN REGISTRATION FUNCTION (I must feed this func)    
        #  uv_rgb = K_rgb * [R | t] * z * inv(K_ir) * uv_ir 
        # unregisteredCameraMatrix	the camera matrix of the depth camera  (depth_cam_matrix)
        # registeredCameraMatrix	the camera matrix of the external camera (uv_cam_matrix)
        # registeredDistCoeffs	the distortion coefficients of the external camera  (infra_cam_dist)
        # Rt	the rigid body transform between the cameras. Transforms points from depth camera frame to external camera frame. (Rt)
        # unregisteredDepth	the input depth data  (depth_image)
        # outputImagePlaneSize the image plane dimensions of the external camera (width, height)
        regDepth = cv.rgbd.registerDepth( depth_cam_matrix, uv_cam_matrix, uv_cam_distort, Rt_to_Uv, depth_image, (resX, resY), depthDilation=False)

        # color_distort = None
        color_regDepth = cv.rgbd.registerDepth( depth_cam_matrix, color_cam_matrix, color_distort, Rt_to_Rgb, depth_image, (resX, resY), depthDilation=False)
        
        regDepth_s =  cv.convertScaleAbs(regDepth, alpha=255/(np.average(regDepth)))
        # aligned_depth_uv = cv.applyColorMap( regDepth_s, cv.COLORMAP_HOT )

        #RGB camera align 
        color_regDepth_s =  cv.convertScaleAbs(color_regDepth, alpha=255/(np.average(color_regDepth)))
        # aligned_depth_rgb = cv.applyColorMap( color_regDepth_s, cv.COLORMAP_HOT )
       
        # FOR DEBUG (only shows the raw colored depth image)    
        # debugo =  cv.convertScaleAbs(depth_image, alpha=255/(np.average(depth_image)))
        # debugo = cv.applyColorMap( debugo, cv.COLORMAP_HOT )
        # cv.imshow('raw depth debug', debugo)

        # cv.circle(aligned_depth_uv, (int(uv_cam_matrix[0][2]),int(uv_cam_matrix[1][2])), 2, (255,0,0), -1)         #DEBUG
        # cv.imshow('aligned depth (uv)', aligned_depth_uv)                                                          #DEBUG
        # cv.circle(aligned_depth_rgb, (int(color_cam_matrix[0][2]),int(color_cam_matrix[1][2])), 2, (255,0,0), -1)  #DEBUG
        # cv.imshow('aligned depth (rgb)', aligned_depth_rgb)                                                        #DEBUG

        # alpha = 0.4
        # aligned_rgb = cv.addWeighted(aligned_depth_rgb, alpha, color_image, 1-alpha, 0.0)
        # aligned_uv = cv.addWeighted(aligned_depth_uv, alpha, infra_image, 1-alpha, 0.0)
        
        # aligned_uv = cv.rectangle(aligned_uv, (vArea[0], vArea[1]), (vArea[0]+vArea[2],vArea[1]+vArea[3]), color=(255,0,0), thickness=5 )
        # M = np.float32([[1.3, 0, 0], [0, 1.3, 0]])
        # aligned_depth_uv = cv.warpAffine(aligned_depth_uv, M, (resX, resY))

        # resized = cv.resize(aligned_depth_uv,None, fx=1.5,fy=1.5, interpolation=cv.INTER_CUBIC)  #scale up UV depth map
        # y = (resized.shape[0]//2) - (resY//2) #- 9 
        # x = (resized.shape[1]//2) - (resX//2) #+ 38
        # cv.imshow('resized', resized)
        # aligned_depth_uv = resized[ y :y + resY, x: x + resX]

        # aligned_uv_rgb = cv.addWeighted(aligned_depth_uv, alpha, aligned_depth_rgb, 1-alpha, 0.0)

        #rescale the aligned depth map and the UV camera image 
        resized = cv.resize(infra_image, None, fx=1.3,fy=1.3, interpolation=cv.INTER_CUBIC)  #scale up UV depth map
        resized = cv.warpAffine(resized, T, (resX,resY))
        regDepth3d = np.dstack((regDepth,regDepth,regDepth)) #depth image is 1 channel, color is 3 channels 
        regDepth3d = cv.resize(regDepth3d, None, fx=1.3,fy=1.3, interpolation=cv.INTER_CUBIC)  #scale up UV depth map
        regDepth3d = cv.warpAffine(regDepth3d, T, (resX,resY))

        #create a mask from the depth map
        # th, mask = cv.threshold(regDepth3d, clipping_distance, 255, cv.THRESH_BINARY_INV)
        # mask = np.where((mask > clipping_distance) | (mask <= 0), 0, 255)  #First version 
        # ai = cv.cvtColor(ai, cv.COLOR_GRAY2BGR)
        # cv.imshow('dbg mask', mask)

        #FOR DEBUG MASK FOR EROTION VALUES
        mask = np.where((regDepth3d > clipping_distance) | (regDepth3d <= 0), 0.0, 255.0)  
        mask = erosion(mask,cv.MORPH_RECT, 5)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = dilatation(mask,cv.MORPH_RECT, 15)  # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE ksize 1~21 
        mask = cv.GaussianBlur( mask,(15 , 15), 0, 0 )
        # mask = np.where((mask > clipping_distance) | (mask <= 0), 0.7, 1.0)
      
        
        #join the uv and the rgb images
        # alpha = np.where((regDepth3d > clipping_distance) | (regDepth3d <= 0), color_image, resized)  #FIRST VERSION
        cv.imshow('mask debug', mask)
        final = blendAlpha(resized, color_image, mask)

        # cv.imshow('aligned to rgb', aligned_rgb)
        # cv.imshow('Aligned to uv', aligned_uv)
        cv.imshow('Aligned UV/RGB', final)
        # cv.imshow('mask debug', mask)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == ord('a'):
            tX = tX - 1.0
            print ("tx:"+ str(tX))
        elif key == ord('d'):
            tX = tX + 1.0
            print ("tx:"+ str(tX))

        #gstream the output
        # video.write(final)       

finally:

    # Stop streaming
    pipeline.stop()



