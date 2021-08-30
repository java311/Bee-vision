###############################################
##      MY OWN OpenCV raw RealSense Align    ##
###############################################

import pyrealsense2 as rs
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import cv2 as cv

resX = 1280
resY = 720

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
depth_sensor.set_option(rs.option.emitter_enabled, True)

# Gets RealSense camera
extrinsics = i_profile.get_extrinsics_to(c_profile)

#Build RS intrinsics in OpenCV matrix format
# K = [fx 0 cx; 
#      0 fy cy; 
#      0  0  1]
depth_cam_matrix = np.zeros((3,3), dtype=np.float)
depth_cam_matrix[0][0] =  intr.fx #fx
depth_cam_matrix[0][2] =  intr.ppx #cx
depth_cam_matrix[1][1] =  intr.fy #fy
depth_cam_matrix[1][2] =  intr.ppy #cy
depth_cam_matrix[2][2] = 1
print ("infrared intrinsics")
print (intr)

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
Rt =  np.zeros((4,4), dtype=np.float32) #rotation matrix

Rt[0][0] = extrinsics.rotation[0]; Rt[0][1] = extrinsics.rotation[1];  Rt[0][2] = extrinsics.rotation[2]; Rt[0][3] = extrinsics.translation[0] 
Rt[1][0] = extrinsics.rotation[3]; Rt[1][1] = extrinsics.rotation[4];  Rt[1][2] = extrinsics.rotation[5]; Rt[1][3] = extrinsics.translation[1] 
Rt[2][0] = extrinsics.rotation[6]; Rt[2][1] = extrinsics.rotation[7];  Rt[2][2] = extrinsics.rotation[8]; Rt[2][3] = extrinsics.translation[2]
Rt[3][0] = 0; Rt[3][1] = 0;  Rt[3][2] = 0; Rt[3][3] = 1  
print (Rt)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # print (depth_image.dtype)
        # depth_image = depth_image * depth_scale   #scale to meters 
        
        # MAIN REGISTRATION FUNCTION (I must feed this func)    
        #  uv_rgb = K_rgb * [R | t] * z * inv(K_ir) * uv_ir 
        # unregisteredCameraMatrix	the camera matrix of the depth camera  (depth_cam_matrix)
        # registeredCameraMatrix	the camera matrix of the external camera (color_cam_matrix)
        # registeredDistCoeffs	the distortion coefficients of the external camera  (color_cam_dist)
        # Rt	the rigid body transform between the cameras. Transforms points from depth camera frame to external camera frame. (Rt)
        # unregisteredDepth	the input depth data  (depth_image)
        # outputImagePlaneSize the image plane dimensions of the external camera (width, height)
        color_distort = None
        regDepth = cv.rgbd.registerDepth( depth_cam_matrix, color_cam_matrix, color_distort, Rt, depth_image, (c_intr.width, c_intr.height), depthDilation=False)
        # plt.imshow(regDepth)
        # plt.show()
        # normal = np.zeros((c_intr.width, c_intr.height))
        # normal = cv.normalize(regDepth, normal, norm_type=cv.NORM_L2)
        regDepth =  cv.convertScaleAbs(regDepth, alpha=255/(np.average(regDepth)))
        # plt.imshow(cmap_reversed)
        # plt.show()
        # print(cmap_reversed)
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

        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())  #This is better 

        # # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        # # Show images
        # cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
        # depth = np.repeat(np.arange(100)[None,:],100,0).astype(float)
        # plt.imshow(regDepth)
        # plt.show()

        alpha = 0.7
        aligned_img = cv.addWeighted(aligned_depth_colormap, alpha, color_image, 1-alpha, 0.0)
        cv.imshow('Aligned Image', aligned_img)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()