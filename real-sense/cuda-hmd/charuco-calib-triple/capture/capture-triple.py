#################################################################
#   CAPTURA USANDO EL RIG DE CAMARAS CHINAS
#   CAMARA STEREO CHINA 
#   CAMARA SENSITIVA PARA UV O INFRAROJO
#   REALSENSE
#################################################################

import cv2
import numpy as np
import pyrealsense2 as rs

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

#prints all the readable camera indices  (TAKES TIME BUT IT IS SIMPLER TO DEBUG)
print (returnCameraIndexes())

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
# 4 RGB LEFT, 5 RGB RIGHT,  1 CENTRAL CAMERA   [0,1,2,4,5]
# 3 RGB LEFT, 2 CENTRAL CAMERA, 4 laptop, 1 RS rgb, 0 RGB RIGHT   [0,1,2,3,4] Connected 1 stereo, center, rs
# 3 RGB LEFT, 2 CENTRAL CAMERA, 0 RGB RIGHT  [0, 1, 2, 3, 4]
right = cv2.VideoCapture(0)  #  right rgb
center = cv2.VideoCapture(2)  # sensor (uv/infrared camera center)
left  = cv2.VideoCapture(3)   # left rgb


#Config RS pipeline
# avaliable unrectified formats:
# https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_D400%20_Custom_Calib_Paper.pdf
points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1920, 1080, rs.format.y16, 15)  #original 25 frames (adjusted accordingly RS docs)  
config.enable_stream(rs.stream.infrared, 2, 1920, 1080, rs.format.y16, 15)  #original 25 frames (adjusted accordingly RS docs) 
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.yuyv, 15)  
profile = pipeline.start(config)

dev = profile.get_device()
depth_sensor = dev.first_depth_sensor()

if (depth_sensor.supports(rs.option.emitter_enabled)):
    depth_sensor.set_option(rs.option.emitter_enabled, False)

# depth_sensor.set_option(rs.option.emitter_enabled, True)

# Check if camera opened successfully
if (left.isOpened() == False or right.isOpened() == False or center.isOpened() == False):
    print("Error opening the stereo rgb cameras or the central camera")

#change webcam resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #original 1280
left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #original 720
right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #original 1280
right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #original 720
center.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #original 1280
center.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #original 720

yuyv_dec = rs.yuy_decoder()  #YUV2RGB realsense conversor

# Read until video is completed
count = 1
print("Press 'c' to capture, or 'q' to quit.")
try:
    while(left.isOpened() and right.isOpened() and center.isOpened() ):
        # Capture frame-by-frame
        ret, left_img = left.read()  #left rgb camera
        ret2, right_img =  right.read() #right rgb camera 
        ret3, center_img = center.read() #uv/infra sensor camera 
        if ret == False and ret2 == False and ret3 == False:
            continue

        frames = pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        # nir_rg_frame = frames.get_infrared_frame(2)
        # rgb_rg_frame = frames.get_color_frame()
        if not nir_lf_frame:
            continue

        nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
        # nir_rg_image =np.asanyarray(nir_rg_frame.get_data())
        # rs_color_image = np.asanyarray(rgb_rg_frame.get_data())

        #rotate the chinese camera images 
        # infracam = cv2.rotate(infracam, cv2.ROTATE_90_CLOCKWISE)
        # uvcam = cv2.rotate(uvcam, cv2.ROTATE_90_CLOCKWISE)

        #Important !! convert the Bayer pattern to RGB !!!
        nir_lf_image = cv2.cvtColor(nir_lf_image, cv2.COLOR_BayerBG2RGB)
        
        #Use the RS filters to transform YUYV to RGB 
        # rgb_rg_frame = yuyv_dec.process(rgb_rg_frame)
        # rs_color_image = np.asanyarray(rgb_rg_frame.get_data())

        #Reduce the resolution of the RS infrared & color images
        nir_lf_image = cv2.resize(nir_lf_image, (1280,720) )
        # rs_color_image = cv2.resize(rs_color_image, (1280,720) )

        #crop RS images (cause the chinese camera is rotated) 
        # leftUp = ( ((1280/2) - 640), ((1080/2) -360) )
        # rightDown =  ( ((720/2) + 640), ((1080/2) + 360) )
        # nir_lf_image = nir_lf_image [0:720, 0:1280]  #(y,x)

        # higher contrast for UV image
        # uvcam = cv2.convertScaleAbs(uvcam, alpha=0.3, beta=0)
        # gray = cv2.cvtColor(uvcam, cv2.COLOR_BGR2GRAY)

        # Display the captured frame
        cv2.imshow('LEFT RGB cam', cv2.resize(left_img, (640,360)) )
        cv2.imshow('RIGHT RGB cam',  cv2.resize(right_img, (640,360)))
        cv2.imshow('CENTER RGB cam',  cv2.resize(center_img, (640,360)))
        cv2.imshow('RS LEFT Infracam', cv2.resize(nir_lf_image, (640,360)))

        #save the images 
        key = cv2.waitKey(23)
        if key & 0xFF == ord(' '):
            cv2.imwrite("captures/cam_left%02d.png" % count, left_img)
            cv2.imwrite("captures/cam_right%02d.png" % count, right_img) 
            cv2.imwrite("captures/cam_center%02d.png"% count, center_img)
            cv2.imwrite("captures/cam_rs_left%02d.png"% count, nir_lf_image)

            print ("Capture no. " + str(count))
            count = count + 1
            print("Press 'SPACE' to capture, or 'q' to quit.")
        elif key & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()

# When everything done, release the video capture object
left.release()
right.release()
center.release()

# Closes all the frames
cv2.destroyAllWindows()

