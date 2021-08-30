##############################################################
##      MY OWN OpenCV align between the UV camera and RS    ##
##############################################################

import pyrealsense2 as rs
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import cv2 as cv

resX = 1280
resY = 720

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

    # T[1] = T[1] + 1.12
    # T[0] = T[0] + 0.12

    fint.release()
    fext.release()

    #format extrinsics for OpenCV use
    Rt[0][0] = R[0][0]; Rt[0][1] = R[0][1];  Rt[0][2] = R[0][2]; Rt[0][3] = T[0]/26.0  
    Rt[1][0] = R[1][0]; Rt[1][1] = R[1][1];  Rt[1][2] = R[1][2]; Rt[1][3] = T[1]/26.0
    Rt[2][0] = R[2][0]; Rt[2][1] = R[2][1];  Rt[2][2] = R[2][2]; Rt[2][3] = T[2]/26.0
    Rt[3][0] = 0; Rt[3][1] = 0;  Rt[3][2] = 0; Rt[3][3] = 1  
    print ("RS depth and UV cam extrinsics")
    print (Rt)

    # Rt[1][3] =  Rt[1][3] + 1.12

    #FOR DEBUG ONLY MANUALLY INSERT 1290x720 M2 MATRIX
    # M2[0][0]=697.8207144; M2[0][1]=0.0; M2[0][2]=663.85083969
    # M2[1][0]=0.0; M2[1][1]=698.70139366; M2[1][2]=365.50826132
    # M2[2][0]=0.0; M2[2][1]=0.0; M2[2][2]=1.0
    # D2 = np.array([ 0.03261677, -0.04987355, -0.0011021,   0.00322985,  0.00468867])


    print ("uv cam intrinsics")
    print (M2)
    return Rt, M2, D2, M1


cap = cv.VideoCapture(3)  #open chinese UV camera 3

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, resX, resY, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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
# c_profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
# c_intr = c_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

dev = cfg.get_device()
depth_sensor = dev.first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, True)

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

# RS extrinsics to OpenCV translation vector and rotation matrix 
Rt =  np.zeros((4,4), dtype=np.float32) #rotation matrix

# Get the extrinsics between the cameras from the calibration YML file 
Rt, infra_cam_matrix, infra_cam_distort, xxx = loadCamFiles('extrinsics.yml', 'intrinsics.yml')

# Check if the UV camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

#change webcam resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH,resX)  
cap.set(cv.CAP_PROP_FRAME_HEIGHT,resY)  

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ret, infra_image = cap.read()  #uv camera
        if not depth_frame  or ret == False:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())
        # print (depth_image.dtype)
        # depth_image = depth_image * depth_scale   #scale to meters 

        # cv.imshow('uv', infra_image)
        # plt.imshow(depth_image)
        # plt.show()

        # For debug !!!
        K_ir = np.array(depth_cam_matrix, copy=True)
        K_ir = np.c_[K_ir, [0,0,0] ]
        K_ir = np.r_[K_ir, [[0,0,0,1]] ]     

        icm = np.array(infra_cam_matrix, copy=True)
        icm = np.c_[icm, [0,0,0] ]
        icm = np.r_[icm, [[0,0,0,1]] ]     

        mDebug1 = np.matmul(Rt, np.linalg.inv(K_ir))
        # print ('mdebug1')
        # print (mDebug1)

        mDebug2 = np.matmul(icm, mDebug1)
        # print 

        
        
        # MAIN REGISTRATION FUNCTION (I must feed this func)    
        #  uv_rgb = K_rgb * [R | t] * z * inv(K_ir) * uv_ir 
        # unregisteredCameraMatrix	the camera matrix of the depth camera  (depth_cam_matrix)
        # registeredCameraMatrix	the camera matrix of the external camera (infra_cam_matrix)
        # registeredDistCoeffs	the distortion coefficients of the external camera  (infra_cam_dist)
        # Rt	the rigid body transform between the cameras. Transforms points from depth camera frame to external camera frame. (Rt)
        # unregisteredDepth	the input depth data  (depth_image)
        # outputImagePlaneSize the image plane dimensions of the external camera (width, height)
        # infra_cam_distort = None
        regDepth = cv.rgbd.registerDepth( depth_cam_matrix, infra_cam_matrix, infra_cam_distort, Rt, depth_image, (resX, resY), depthDilation=False)
        # plt.imshow(regDepth)
        # plt.show()
        # normal = np.zeros((c_intr.width, c_intr.height))
        # normal = cv.normalize(regDepth, normal, norm_type=cv.NORM_L2)
        regDepth =  cv.convertScaleAbs(regDepth, alpha=255/(np.average(regDepth)))
       
        aligned_depth_colormap = cv.applyColorMap( regDepth, cv.COLORMAP_HOT )
       
        # #DEBUG
        # d_mat1 = np.eye(4)
        # for i in range(0,len(depth_cam_matrix),1):
        #     for j in range(0,len(depth_cam_matrix),1):
        #         d_mat1[i][j] = depth_cam_matrix[i][j]

        # first_proj = Rt.dot(np.linalg.pinv(d_mat1))
        # print(first_proj)

        
        # break
        

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # aligned_depth_colormap = cv.applyColorMap(cv.convertScaleAbs(regDepth, alpha=0.9), cv.COLORMAP_JET) #Not good enough

        # depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())  #This is better 

        # # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        # # Show images
        # cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
        # depth = np.repeat(np.arange(100)[None,:],100,0).astype(float)
        # plt.imshow(regDepth)
        # plt.show()

       
        debugo =  cv.convertScaleAbs(depth_image, alpha=255/(np.average(depth_image)))
        debugo = cv.applyColorMap( debugo, cv.COLORMAP_HOT )
        cv.imshow('depth', debugo)
        cv.imshow('aligned color map', aligned_depth_colormap)

        alpha = 0.7
        aligned_img = cv.addWeighted(aligned_depth_colormap, alpha, infra_image, 1-alpha, 0.0)
        cv.imshow('Aligned Image', aligned_img)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()