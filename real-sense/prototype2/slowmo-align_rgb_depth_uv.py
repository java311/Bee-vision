import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt  #matplotlib for DEBUG only

#PROGRAM CONSTANTS 
mapx1=None; mapx2=None; mapy1=None; mapy2=None
alphaValue = 0.6  #alpha for the mask transparency
resX = 1280
resY = 720

#This is a faster compositor
def fastAlphaBlend(fg,bg,alpha):
    # MY VERSION
    a = alpha[:, :, np.newaxis]
    blended = cv2.convertScaleAbs(fg * a + bg * (1-a))
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

    # def undistort(M1, M2, D1, D2, R1, R2, P1, P2):
    undistort(M1, M2, D1, D2, R1, R2, P1, P2) 

#rectifies the stereo camera pair
def remap(left, right, depth, mask ):  #left is rgb_RS, right is UV, depth is UV depth 
    #remap(mg, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR)
    recLeft = np.full_like(left, 0, dtype="uint8")
    recRight = np.full_like(right, 0, dtype="uint8")
    recDepth = np.full_like(depth, 0, dtype="float")
    recMask = np.full_like(mask, 0, dtype="float")

    recLeft  = cv2.remap(left,  mapx1, mapy1, cv2.INTER_LINEAR)  #rgb
    recRight = cv2.remap(right, mapx2, mapy2, cv2.INTER_LINEAR) #uv
    recDepth = cv2.remap(depth, mapx2, mapy2, cv2.INTER_LINEAR) #depth 
    recMask = cv2.remap(mask, mapx2, mapy2, cv2.INTER_LINEAR) #mask
    #recRight = cv2.warpPerspective(recRight, self.R1, (self.width, self.height))

    return recLeft, recRight, recDepth, recMask
    # return self.cutImage(recLeft, recRight)

#Disparity from depthMap (depthMap should be given in meters)
baseLineY = 22.780315216227873  #taken from extrinsics T vector [ -1.1517719701854450e+01, 2.3134096167196148e+01,-1.0639161017443953e+01 ]
fy = 808.01107745281718  # taken from extrinsics  6.9519449163456534e+02  8.0002217022260618e+02,
def getDisparityMap(depthMap):
    base = baseLineY * fy 
    dispMap = np.zeros((resY,resX), dtype=np.float)
    # plt.imshow(depthMap)
    # plt.show()

    for y in range(0, resY, 1):
        for x in range(0, resX, 1):
            if (3 < depthMap[y][x] < 3000):
                dispMap[y][x] = base / depthMap[y][x]  #in pixels (in theory ) disp in Y
    return dispMap
    


def rectifyUV(back, front, mask, dispMap):
    for x in range(0, resX, 1):
        for y in range(0, resY, 1):
            if ( y + int(round(dispMap[y][x])) < resY) and mask[y][x] > 0:
                back[y][x][0] = front[y + int(round(dispMap[y][x]))] [x][0]  
                back[y][x][1] = front[y + int(round(dispMap[y][x]))] [x][1]
                back[y][x][2] = front[y + int(round(dispMap[y][x]))] [x][2]      
    return back


# Create a pipeline
pipeline = rs.pipeline()
cap = cv2.VideoCapture(3)  #open chinese UV camera 3

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
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.0 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
print("Clipping Distance is: " , clipping_distance)

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

loadCamFiles('./rgb_uv_params/extrinsics.yml', './rgb_uv_params/intrinsics.yml')
# Streaming loop
try:
    while True:
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

        # Remove background - Set pixels further than clipping_distance to grey
        # grey_color = 153
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        mask = np.where((depth_image > clipping_distance) | (depth_image <= 0.0), 0.0, alphaValue)

        disp_image = getDisparityMap(depth_image)  #Transforms the depth map to disparity map
        rec_color_image, rec_uv_image, rec_disp_image, rec_mask = remap(color_image, uv_image, disp_image, mask)  #Rectify UV, RGB and depth

        final = rectifyUV(rec_color_image, rec_uv_image, rec_mask, rec_disp_image)

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

        # mask = np.full((resY, resX), alphaValue) #DEBUG
        # final = fastAlphaBlend(rec_uv_image, rec_color_image, mask)

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
        

        images = np.hstack((final, rec_uv_image))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
