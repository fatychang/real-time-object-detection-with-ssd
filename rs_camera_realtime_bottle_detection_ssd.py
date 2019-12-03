# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:44:56 2019

This script tracks the bottle object in the view and 
returns the distance from the camera to the object.

It implements the ssd and mobile network which is trained by COCO dataset.
Other pre-trained classes are filtered out in this script.
Only bottle will be detected.

@author: jschang
"""

#import necessary packages
import numpy as np
import pyrealsense2 as rs
import cv2
import pptk

# add sys path for self-build packages
import sys
sys.path.append('D:\Jen\Projects\RealSense Camera\Codes\Python\selfbuildpackages')

# import selfbuildpackages
from network_ssd import SSD



# configure the realsense camera
pipeline = rs.pipeline()
config = rs.config()

# set the video dimensions and frame rate
width = 640
height = 480
fr = 30

# enable streams
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fr)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fr)


# create an align object
# rs.aligh allows us to perform alignement of depth to other fames
# the align_to is the stream to which we plan to aligh depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# start streaming
profile = pipeline.start(config)

# obtain depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("[INFO] depth scale: {:.5f}".format(depth_scale))

# declare pointcloud instance
pc = rs.pointcloud()


print("[INFO] camera initailization done.")


# declare ssd network instance
ssd = SSD().start()
ssd.load_model()
IGNORED = ["background", "aeroplane", "bicycle", "bird", "boat",
            	"bus", "car", "cat", "chair", "cow", "diningtable",
            	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            	"sofa", "train", "tvmonitor"]

ssd.set_ignored_classes(IGNORED)

# load classes and colors lisk
CLASSES = ssd.load_classes()
COLORS = ssd.generate_colors(len(CLASSES))
print("[INFO] network model loaded ready.")


print("[INFO] Start streaming...")
# loop over frames from the streams
try:
    while True:
        # wait for a coherent pair of frames: color and depth frame
        frames = pipeline.wait_for_frames()
    
        # align the depth frame to color frame
        aligned_frames = align.process(frames)    
        
        # get the aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_depth_frame
        if not depth_frame or not color_frame:
            continue
        
        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # map pointcloud to depth frame
        points = pc.calculate(depth_frame)
        pc.map_to(depth_frame)
        
        # point cloud data to numpy array
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asarray(v).view(np.float32).reshape(-1,3)        #xyz
        verts = np.asarray(v).view(np.float32).reshape(480, 640 ,3)   
        texcoords = np.asarray(t).view(np.float32).reshape(-1,2)    #uv 
        
        
        # blob the image and get the dimension of the image
        (h, w) = color_image.shape[:2]
        blob = ssd.blob_image(color_image)
        
        
        # obtain detections
        detections = ssd.get_detections(blob)
        
        for i in range (0, len(detections)):
            # obtain the results
            confidence = ssd.get_confidence(detections, i)
            
            # filter out the weak confidnece class
            if confidence > 0.5:
                # obtain the class idx. label and corresponding color
                class_idx = ssd.get_class_idx(detections, i)
                class_label = ssd.get_class_label(class_idx)
                color = ssd.get_color(class_idx)
                
                # skip the detection if it fits the ignored list
                if class_label in IGNORED:
                    continue
                
                # obtain the bounding box
                box = ssd.get_bounding_box(detections, i)
                startX, startY, endX, endY = box
                
                # draw the bounding box and result text
                ssd.draw_bounding_box(color_image, detections, i, color)
                
                
                # get the position of the box center
                box_center = ssd.get_box_center()
                # get the distance of the center
                center_dist = depth_frame.get_distance(box_center[0], box_center[1])
                # update center distance to the image
                text_cen = "Dis:{:.2f}".format(center_dist)
                cv2.putText(color_image, text_cen, (box_center[0], box_center[1]-15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)  
                
                # obtain the bounding box depth
                box_z = verts[startY:endY, startX:endX, :]
                
        ###
        # Visualization
        cv2.imshow("video", color_image)
        key = cv2.waitKey(1) & 0xFF
        
#        # visulaizae (FOR DEBUG ONLY)
#        viewer = pptk.viewer(verts)
#        viewer.set(point_size=0.005)
            
        
        # quit the loop by pressing 'q'
        if key == ord("q"):
            break
finally:    
    # clear out
    pipeline.stop()
    cv2.destroyAllWindows()               
    
    