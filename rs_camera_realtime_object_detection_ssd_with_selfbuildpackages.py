# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:57:40 2019

@author: jschang
"""

# import sys model to appending paths
import sys
sys.path.append('D:\Jen\Projects\RealSense Camera\Codes\Python\selfbuildpackages')
# import self-build packages
from network_ssd import SSD
# import realsense package
import pyrealsense2 as rs
# import opencv package
import cv2
# import package for calculate fps

#import other packages
import numpy as np
import time


dir_coffe_prototxt = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\real-time-object-detection\\MobileNetSSD_deploy.prototxt.txt"
dir_coffe_model = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\real-time-object-detection\\MobileNetSSD_deploy.caffemodel"
confidence_thresh = 0.5


# initialize the model
ssd = SSD().start()
ssd.load_model(dir_coffe_prototxt, dir_coffe_model)
print("[INFO] network model loaded.")
# load the classes
CLASSES = ssd.load_classes()
# generate color list for the classes
COLORS = ssd.generate_colors(len(CLASSES))


# config realsense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640 ,480, rs.format.bgr8, 30)




# create an align object
# rs.aligh allows us to perform alignement of depth to other fames
# the align_to is the stream to which we plan to aligh depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# start the rs-camera stream
profile = pipeline.start(config)
# sleep to worn up the camera
time.sleep(2.0)


# obtain the depth sensor and its depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("[INFO] Depth scale is {}".format(depth_scale))



print("[INFO] Start streaming...")
# loop over frames from the streams
try:
    while True:
        
        # wait for a coherent pair of frames: color and depth frame
        frames = pipeline.wait_for_frames()
        
        # align the depth frame to color frame
        aligned_frames = align.process(frames)

#        # get the frames        
#        depth_frame = frames.get_depth_frame()
#        color_frame = frames.get_color_frame()
        
        # get the aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_depth_frame
        
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue 
        
        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # depth image 3d. (depth image is 1-channel, color is 3 channels)
        depth_image_3d = np.dstack((depth_image*depth_scale,depth_image*depth_scale
                                    , depth_image*depth_scale))
        
       
        
        
        # apply color map to the depth image
        # image must converted to 8-bit per pixel first
        depth_colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                               alpha=0.03), cv2.COLORMAP_JET)
        
        # blob the image and get the dimension of the image
        (h, w) = color_image.shape[:2]
        blob = ssd.blob_image(color_image)
        
        # obtain detections
        detections = ssd.get_detections(blob)
        
        # initialize lists
        box_center=[]
        
        # loop over the detections
        for i in range (0, len(detections)):
            
            # get the idx and color
            idx = ssd.get_class_idx(detections, i)
            color = ssd.get_color(COLORS, idx)
            
            # draw the bounding boxes
            ssd.draw_bounding_box(color_image, detections, i, COLORS)
            
            # get the position of the box center
            box_center = ssd.get_box_center()
            
            # get the distance of the center
            center_dist = depth_frame.get_distance(box_center[0], box_center[1])
            # update center distance to the image
            text_cen = "Dis:{:.2f}".format(center_dist)
            cv2.putText(color_image, text_cen, (box_center[0], box_center[1]-15),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)         

        
        # stack both image together
        images = np.hstack((color_image, depth_colormap_image))
        
        
        # show the image
        cv2.namedWindow("Realsense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Realsense", images)
        key = cv2.waitKey(1) & 0xFF
        
        # quit the loop by pressing 'q'
        if key == ord("q"):
            break
        
        
finally:
    pipeline.stop()
    cv2.destroyAllWindows()