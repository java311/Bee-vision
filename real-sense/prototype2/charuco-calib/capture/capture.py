import cv2
import numpy as np
import pyrealsense2 as rs


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(3)
# cap2 = cv2.VideoCapture(2)  

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
if (cap.isOpened() == False):
    print("Error opening video stream or file")

#change webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #original 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #original 720

yuyv_dec = rs.yuy_decoder()  #YUV2RGB realsense conversor

# Read until video is completed
count = 1
print("Press 'c' to capture, or 'q' to quit.")
try:
    while(cap.isOpened() ):
        # Capture frame-by-frame
        ret, infracam = cap.read()
        if ret == False:
            continue

        frames = pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        nir_rg_frame = frames.get_infrared_frame(2)
        rgb_rg_frame = frames.get_color_frame()
        if not nir_lf_frame or not nir_rg_frame:
            continue

        nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
        nir_rg_image =np.asanyarray(nir_rg_frame.get_data())
        # rs_color_image = np.asanyarray(rgb_rg_frame.get_data())

        #Important !! convert the Bayer pattern to RGB !!!
        nir_lf_image = cv2.cvtColor(nir_lf_image, cv2.COLOR_BayerBG2RGB)
        
        #Use the RS filters to transform YUYV to RGB 
        rgb_rg_frame = yuyv_dec.process(rgb_rg_frame)
        rs_color_image = np.asanyarray(rgb_rg_frame.get_data())

        #Reduce the resolution of the RS infrared & color images
        nir_lf_image = cv2.resize(nir_lf_image, (1280,720) )
        rs_color_image = cv2.resize(rs_color_image, (1280,720) )

        #Crop the resolution from the RS 1920x1080 to 1280x720
        # leftUp = ( ((1920/2) - 640), ((1080/2) -360) )
        # rightDown =  ( ((1920/2) + 640), ((1080/2) + 360) )
        # nir_lf_image = nir_lf_image [0:720, 0:1280]  #(y,x)

        # Display the captured frame
        cv2.imshow('RS left infra', nir_lf_image)
        cv2.imshow('UV cam',infracam)
        cv2.imshow('RGB RS', rs_color_image)

        #save the images 
        key = cv2.waitKey(23)
        if key & 0xFF == ord(' '):
            cv2.imwrite("captures/cam_uv%02d.png" % count, infracam)
            # cv2.imwrite("captures/rs_right%02d.png" % count, nir_rg_image)
            cv2.imwrite("captures/cam_in%02d.png" % count, nir_lf_image) 
            cv2.imwrite("captures/cam_rgb%02d.png"% count, rs_color_image)

            print ("Capture no. " + str(count))
            count = count + 1
            print("Press 'SPACE' to capture, or 'q' to quit.")
        elif key & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

