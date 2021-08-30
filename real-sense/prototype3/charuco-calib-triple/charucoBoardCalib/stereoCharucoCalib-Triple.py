import numpy
import cv2
from cv2 import aruco
import os
import glob

# -w=6 -h=8 -sl=350 -ml=225 -d=10 "chboard.png" MY charuco board parameters
# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 8
CHARUCOBOARD_COLCOUNT = 6 
CHARUCODICT_ID = 10
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
PATH = 'C:\\bee-vision\\source\\real-sense\\prototype3\\charuco-calib-triple\\capture\\captures_triple_9\\'
CAM1_MASK = 'cam_rgb*.png'
CAM2_MASK = 'cam_uv*.png'
RESULT_PREF = 'rgb_uv'
RES_X = 1280
RES_Y = 720

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=31, #33 original #31 lab   
        markerLength=20, #21 original #20 lab
        dictionary=ARUCO_DICT)

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners1_all = [] # Corners discovered in all images processed
ids1_all = [] # Aruco ids corresponding to corners discovered
corners2_all = [] # Corners discovered in all images processed
ids2_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime

# DEBUG DRAW the charuco board
margins = 0.031 - 0.020  #0.31 - 0.020 (lab)  #0.033 - 0.021 (original)   
boardImage = CHARUCO_BOARD.draw((595,842), margins, 0)  #original (720,1280)
cv2.imwrite('charuco.png', boardImage)
# cv2.imshow('Charuco', boardImage)
# cv2.waitKey(0)

# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
cam1_images = glob.glob(PATH + CAM1_MASK)
cam2_images = glob.glob(PATH + CAM2_MASK)
# images = glob.glob('C:\bee-vision\source\real-sense\capture\captures_graymount_charuco_1\cam_uv*.png')

if ((len(cam1_images) != len(cam2_images)) or  ( len(cam1_images) < 1 or len(cam2_images) < 1) ):
    print ("First camera images: " + str(len(cam1_images)))
    print ("Second camera images: " + str(len(cam2_images)))
    print("ERROR: both cameras should have the same image size")
    exit()


def getCharucoIds(imagePath):
    # Open the images
    img = cv2.imread(imagePath)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find aruco markers in the query image
    corners, ids, _ = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

    response = 0
    if (ids is not None):
        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=CHARUCO_BOARD)
    else:
        print ("Corners not found...")
        return 0, None, None

    #Show the detected corners if found
    if response > 10:
        # Outline the aruco markers found in our query image
        img = aruco.drawDetectedMarkers(
                image=img, 
                corners=corners)

        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)

        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
        # Pause to display each image, waiting for key press
        cv2.imshow('Charuco board', img)
        cv2.waitKey(1)

    return response, charuco_corners, charuco_ids


# Loop through images glob'ed
for i in range(0,len(cam1_images),1):
    cam1_iname = cam1_images[i]
    cam2_iname = cam2_images[i]

    # print (cam1_iname)
    # print (cam2_iname)

    response1, charuco_corners1, charuco_ids1 = getCharucoIds(cam1_iname)
    response2, charuco_corners2, charuco_ids2 = getCharucoIds(cam2_iname)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 30 squares
    if (response1 + response2 > 20 and response1 == response2):  #original 30
        # Add these corners and ids to our calibration arrays
        corners1_all.append(charuco_corners1)
        ids1_all.append(charuco_ids1)
        corners2_all.append(charuco_corners2)
        ids2_all.append(charuco_ids2)
    else:
        # print("Not able to detect a charuco board in image: {}".format(cam1_images[i]))
        print("R1: "+ str(response1) + " R2: " +str(response2)+ " : " + format(cam1_images[i]) ) 

print ("Final image pairs used: " + str(len(corners1_all)) + ":" + str(len(corners2_all)) )

# Destroy any open CV windows
cv2.destroyAllWindows()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
print ("Performing calibration of camera 1 wait....")
retval_1, cameraMatrix_1, distCoeffs_1, rvecs_1, tvecs_1 = aruco.calibrateCameraCharuco(
        charucoCorners=corners1_all,
        charucoIds=ids1_all,
        board=CHARUCO_BOARD,
        imageSize=(RES_X,RES_Y),
        cameraMatrix=None,
        distCoeffs=None)
    
# Print matrix and distortion coefficient to the console
print("CAM 1 RMS: " + str(retval_1))
print(cameraMatrix_1)
print(distCoeffs_1)
print ("Performing calibration of camera 2 wait....")
retval_2, cameraMatrix_2, distCoeffs_2, rvecs_2, tvecs_2 = aruco.calibrateCameraCharuco(
        charucoCorners=corners2_all,
        charucoIds=ids2_all,
        board=CHARUCO_BOARD,
        imageSize=(RES_Y,RES_X),
        cameraMatrix=None,
        distCoeffs=None)
    
# Print matrix and distortion coefficient to the console
print("CAM 2 RMS: " + str(retval_2))
print(cameraMatrix_2)
print(distCoeffs_2)

# TODO load the intrinsics from a file 

#Finally perform the Stereo calibration
objectPoints = []
for i in range(0,len(corners1_all),1):
    objectPoints.append(CHARUCO_BOARD.chessboardCorners)


stereocalibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalibration_flags = ( #cv2.CALIB_FIX_ASPECT_RATIO +
                          cv2.CALIB_USE_INTRINSIC_GUESS +
                        #   cv2.CALIB_FIX_INTRINSIC +
                          cv2.CALIB_ZERO_TANGENT_DIST +
                          cv2.CALIB_RATIONAL_MODEL +
                          cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5)
rms, camMat1, dCoeffs1, camMat2, dCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints,corners1_all,corners2_all,  cameraMatrix_1, distCoeffs_1, cameraMatrix_2, distCoeffs_2, (RES_X,RES_Y), criteria = stereocalibration_criteria, flags = stereocalibration_flags)
print ("STEREO RMS:" +str(rms))

# stereocalibration_flags = ( cv2.CALIB_FIX_ASPECT_RATIO +
#                         #   cv2.CALIB_USE_INTRINSIC_GUESS +
#                            cv2.CALIB_FIX_INTRINSIC +
#                           cv2.CALIB_ZERO_TANGENT_DIST +
#                           cv2.CALIB_FIX_PRINCIPAL_POINT )
# rms, camMat1, dCoeffs1, camMat2, dCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints,corners1_all,corners2_all,  cameraMatrix_1, distCoeffs_1, cameraMatrix_2, distCoeffs_2, (RES_X,RES_Y), flags = stereocalibration_flags)


#save intrinsic matrices of both cameras
filename = PATH + RESULT_PREF + '_intrinsics.yml'
fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
fs.write(name='M1',val=camMat1)
fs.write(name='D1',val=dCoeffs1)
fs.write(name='M2',val=camMat2)
fs.write(name='D2',val=dCoeffs2)
fs.release()

#Calculate the rectification matrices 
R1, R2, P1, P2, Q, vRoi1, vRoi2 = cv2.stereoRectify(camMat1, dCoeffs1,camMat2, dCoeffs2, (RES_X, RES_Y), R, T,cv2.CALIB_ZERO_DISPARITY, 1, (RES_X, RES_Y))  #original 1,  (RES_X, RES_Y)

#save extrinsics matrices of both cameras 
filename = PATH + RESULT_PREF + '_extrinsics.yml'
fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
fs.write(name='R',val=R)
fs.write(name='T',val=T)
fs.write(name='R1',val=R1) 
fs.write(name='R2',val=R2)
fs.write(name='P1',val=P1) 
fs.write(name='P2',val=P2)
fs.write(name='Q',val=Q)
fs.write(name='validRoi1',val=vRoi1)
fs.write(name='validRoi2',val=vRoi2)
fs.release()

#image rectification test 
print("Image rectification test")

mapx1 = numpy.ndarray(shape=(RES_Y, RES_X, 1), dtype='float32')  #ORIGINAL RES_Y, RES_X
mapy1 = numpy.ndarray(shape=(RES_Y, RES_X, 1), dtype='float32')
mapx2 = numpy.ndarray(shape=(RES_Y, RES_X, 1), dtype='float32')
mapy2 = numpy.ndarray(shape=(RES_Y, RES_X, 1), dtype='float32')

mapx1, mapy1 = cv2.initUndistortRectifyMap(camMat1, dCoeffs1, R1, P1,(RES_X, RES_Y), cv2.CV_16SC2)  #original (RES_X, RES_Y)
mapx2, mapy2 = cv2.initUndistortRectifyMap(camMat2, dCoeffs2, R2, P2,(RES_X, RES_Y), cv2.CV_16SC2)


folder = PATH + 'rectify_' + RESULT_PREF + '\\'  #make directory for the rectify output
if not os.path.exists(folder):
    os.mkdir(folder)
for i in range(0,len(cam1_images),1):
    down = cv2.imread(cam1_images[i])
    up = cv2.imread(cam2_images[i])

    recDown = numpy.full_like(down, 0, dtype="uint8")
    recUp = numpy.full_like(up, 0, dtype="uint8")
    #remap the images 
    recUp = cv2.remap(up, mapx2, mapy2, cv2.INTER_LINEAR)
    recDown = cv2.remap(down, mapx1, mapy1, cv2.INTER_LINEAR) 
    #paint the rectangles
    recUp = cv2.rectangle(recUp, (vRoi1[0], vRoi1[1]) , (vRoi1[2], vRoi1[3]), (0, 0, 255),2)
    recDown = cv2.rectangle(recDown, (vRoi2[0], vRoi2[1]) , (vRoi2[2], vRoi2[3]), (0, 0, 255),2)
    #join the images
    final = cv2.vconcat([recDown, recUp])   
    # final = cv2.resize(final, (RES_X//2, RES_Y//2) )  #only added to fit in the screen
    # cv2.imshow('Final Result - ' + str(i), final)
    cv2.imwrite(folder + '\\rectified_'+str(i)+'.png', final)
    cv2.waitKey(1)


    
# Save values to be used where matrix+dist is required, for instance for posture estimation
# I save files in a pickle file, but you can use yaml or whatever works for you
# f = open(filename, 'wb')
# yaml.dump((retval, cameraMatrix, distCoeffs, rvecs, tvecs), f)
# f.close()

# filename = MASK.replace('*', '')
# filename = filename.replace('.png' ,'.yml')
# filefullpath = PATH + 'instrinsics_' + filename 

# fs = cv2.FileStorage(filefullpath, cv2.FILE_STORAGE_WRITE)
# fs.write(name='RMS',val=retval)
# fs.write(name='cameraMatrix',val=cameraMatrix)
# fs.write(name='distCoeffs',val=distCoeffs)
# fs.release()
    
# # Print to console our success
# print('Calibration successful. Calibration file used: '+ filefullpath)

