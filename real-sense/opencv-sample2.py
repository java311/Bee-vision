import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os


def extract_from_bag(bag_fname, color_fname, depth_fname):

    config = rs.config()
    pipeline = rs.pipeline()

    # make it so the stream does not continue looping
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    rs.config.enable_device_from_file(config, bag_fname, repeat_playback=False)
    profile = pipeline.start(config)
    # this makes it so no frames are dropped while writing video
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()

    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_matrices = []

    i = 0
    while True:

        # when stream is finished, RuntimeError is raised, hence this
        # exception block to capture this
        try:
            # frames = pipeline.wait_for_frames()
            frames = pipeline.wait_for_frames(timeout_ms=100)
            if frames.size() <2:
                # Inputs are not ready yet
                continue
        except (RuntimeError):
            print('frame count', i-1)
            pipeline.stop()
            break

        # align the deph to color frame
        aligned_frames = align.process(frames)

        # get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        scaled_depth_image = depth_image * depth_scale
        color_image = np.asanyarray(color_frame.get_data())

        # convert color image to BGR for OpenCV
        r, g, b = cv2.split(color_image)
        color_image = cv2.merge((b, g, r))

        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('Aligned Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Aligned Example', images)

        fname = "frame{:06d}".format(i) + ".png"
        cv2.imwrite(color_fname + fname, color_image)

        depth_matrices.append(scaled_depth_image)

        # color_out.write(color_image)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        i += 1

    # release everything now that job finished
    np.save(depth_fname, np.array(depth_matrices))
    print("Size of depth matrices:", len(depth_matrices))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help=".bag file to read")
    parser.add_argument("-c", "--rgbfilename", type=str, help=".mp4 file to save RGB stream")
    parser.add_argument("-d", "--depthfilename", type=str, help=".npy file to save depth stream")
    args = parser.parse_args()

    extract_from_bag(bag_fname=args.input, color_fname=args.rgbfilename, depth_fname=args.depthfilename)