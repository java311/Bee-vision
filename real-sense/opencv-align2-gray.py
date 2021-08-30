
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

colorizer = rs.colorizer()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        #Aligned images 
        depth_image_aligned = np.asanyarray(aligned_depth_frame.get_data())

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Stack normal images horizontally
        print (depth_image.dtype)
        depth_gray_image = np.zeros((480,640,3), dtype=np.uint16)
        print (depth_gray_image.dtype)
        # depth_gray_image = cv2.merge((depth_image, depth_image, depth_image))
        # depth_gray_image = np.zeros_like(img)
        depth_gray_image[:,:,0] = depth_image
        depth_gray_image[:,:,1] = depth_image
        depth_gray_image[:,:,2] = depth_image
        cv2.imwrite('debugsito.png', depth_gray_image)
        images = np.hstack((color_image, depth_gray_image))

        # Render ALIGNED images
        # depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_aligned = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        alpha = 0.7
        aligned_img = cv2.addWeighted(depth_colormap_aligned, alpha, color_image, 1-alpha, 0.0)
        cv2.imshow('Aligned Example', aligned_img)

        # Show normal images
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()