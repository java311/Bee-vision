import cv2
import time
import numpy as np

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

#print readable camera indices
print (returnCameraIndexes())

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(1)   #chinese camera 1
cap2 = cv2.VideoCapture(2)  #chinese camera 2  
cap3 = cv2.VideoCapture(3)  #chinese camera 3
time.sleep(5)

# Check if camera opened successfully
if (cap.isOpened() == False or cap2.isOpened() == False or cap3.isOpened() == False):
    print("Error opening video stream or file")

#change webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #original 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #original 720
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #original 1280
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #original 720
cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #original 1280
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #original 720

# Read until video is completed
count = 1
print("Press 'SPACE' to capture, or 'q' to quit.")
try:
    while(cap.isOpened() and cap.isOpened() ):
        # Capture frame-by-frame
        ret, img1 = cap.read()  #cam1
        ret2, img2 =  cap2.read() #cam2
        ret3, img3 =  cap3.read() #cam3 
        if ret == False and ret2 == False and ret3 == False:
            continue

        # Display the captured frame
        cv2.imshow('CAM 1', img1)
        cv2.imshow('CAM 2', img2)
        cv2.imshow('CAM 3', img3)
        # cv2.imshow('RGB RS',  rs_color_image)

        #save the images 
        key = cv2.waitKey(23)
        if key & 0xFF == ord(' '):
            cv2.imwrite("captures/cam_01_%02d.png" % count, img1)
            cv2.imwrite("captures/cam_02_%02d.png" % count, img2)
            cv2.imwrite("captures/cam_03_%02d.png" % count, img3) 
            # cv2.imwrite("captures/cam_rgb%02d.png"% count, rs_color_image)

            print ("Capture no. " + str(count))
            count = count + 1
            print("Press 'SPACE' to capture, or 'q' to quit.")
        elif key & 0xFF == ord('q'):
            break
finally:

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

