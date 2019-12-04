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
from PointCloudDrawer import AppState



##############################################################################
## 0. Declaration params
isPause = 0



###############################################################################
## 1. Initialize camera

# declare pipeline and config instance
pipeline = rs.pipeline()
config = rs.config()

# declare pointcloud instance
pc = rs.pointcloud()
# declare colorizer instance
colorizer = rs.colorizer()

# set the video dimensions and frame rate
width = 640
height = 480
fr = 30

## find realsense device
#ctx = rs.context()
#devices = ctx.query_devices()
#print("[INFO] Devices found {}".format(devices))
#dev = devices[0]
#
## enable device advanced mode
#advnc_mode = rs.rs400_advanced_mode(dev)
#print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disable")
#
## reduce the max depth to 1.5 meters
#depth_table = advnc_mode.get_depth_table()
#depth_table.depthClampMax = 1500
#advnc_mode.set_depth_table(depth_table)


# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, "test.bag")

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
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
# obtain depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("[INFO] depth scale: {:.5f}".format(depth_scale))
print("[INFO] camera initailization done.")




###############################################################################
## 2. Initialize network 

# declare ssd network instance
ssd = SSD().start()
ssd.load_model()

# defined ignored list from the COCO dataset
# here we only detect bottle object
IGNORED = ["background", "aeroplane", "bicycle", "bird", "boat",
            	"bus", "car", "cat", "chair", "cow", "diningtable",
            	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            	"sofa", "train", "tvmonitor"]
ssd.set_ignored_classes(IGNORED)

# load classes and colors lisk
CLASSES = ssd.load_classes()
COLORS = ssd.generate_colors(len(CLASSES))
print("[INFO] network model loaded.")



###############################################################################
##  3. Initialize OpenCV windows and pointcloud viewer

# declare pointcloud state instance
pc_state = AppState()

# declare a numpy array stores the pointcloud data for visualization
out = np.empty((height, width,3), dtype=np.uint8)
pc_state.get_size(out)

# opencv window
cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Pointcloud", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Pointcloud", pc_state.mouse_cb)





###############################################################################
##  4. start streaming

print("[INFO] Start streaming...")
# loop over frames from the streams
try:
    while True:
        if not isPause:
            
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
            
            # Apply color map to depth image (image must converted to 8-bit per pixel first)
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # apply color map to the depth image
            # image must converted to 8-bit per pixel first
            depth_colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                                   alpha=0.03), cv2.COLORMAP_JET)
            
            # map pointcloud to depth frame
            points = pc.calculate(depth_frame)
            pc.map_to(depth_frame)
            
            # point cloud data to numpy array
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asarray(v).view(np.float32).reshape(-1,3)        #xyz 
            texcoords = np.asarray(t).view(np.float32).reshape(-1,2)    #uv 
            
            # pointcloud data to numpy remains the shape (3D)
            verts_image = np.asarray(v).view(np.float32).reshape(height, width, 3) 
            
            
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
                    box_z = verts_image[startY:endY, startX:endX, :]
                    box_z_verts = box_z.reshape(-1, 3)
                    
                    # display position (x, y, z)
                    # noted: the verts_image stores the elements in (row, column),
                    # therefor, box_center[1] equals to row and
                    # box_center[2] equals to column
                    text_cen_x = "Center x:{:.2f}".format(verts_image[box_center[1], box_center[0],0])
                    text_cen_y = "Center y:{:.2f}".format(verts_image[box_center[1], box_center[0],1])
                    text_cen_z = "Center z:{:.2f}".format(verts_image[box_center[1], box_center[0],2])
                    cv2.putText(color_image, text_cen_x, (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2) 
                    cv2.putText(color_image, text_cen_y, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                    cv2.putText(color_image, text_cen_z, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                    
            #######################################################################
            # 5.  Visualization
            
            # render pointcloud
            out.fill(0)
            pc_state.grid(out, (0, 0.5, 1), size=1, n=10)
            pc_state.frustum(out, depth_intrinsics)
            pc_state.axes(out, pc_state.view([0,0,0]), pc_state.rotation, size=0.1, thickness=1)
            
            pc_state.pointcloud_display(out, box_z_verts, texcoords, depth_colormap)
            
            # clear the verts
            box_z_verts.fill(0)
            
            # monitoring mouse buttons
            if any(pc_state.mouse_btns):
                pc_state.axes(out, pc_state.view(pc_state.pivot), pc_state.rotation, thickness=4)
            
            cv2.imshow("RGB", color_image)
            cv2.imshow("Depth", depth_colormap_image)
            cv2.imshow("Pointcloud", out)
        
        
        
        
        key = cv2.waitKey(1) & 0xFF
        
        # visulaizae (FOR DEBUG ONLY)
#        viewer = pptk.viewer(box_z_verts)
#        viewer.set(point_size=0.005)
            
        
        # quit the loop by pressing 'q' or 'esc'
        if key == ord("q") or key == 27:
            break
        # pause the frame when pressing 'p'
        if key == ord("p"):
            isPause = not isPause
finally:    
    # clear out
    pipeline.stop()
    cv2.destroyAllWindows()  
#    viewer.close()            
    
    