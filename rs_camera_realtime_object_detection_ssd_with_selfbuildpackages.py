# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:57:40 2019

This script performs visualization of the real-time object detection.
It shows the RGB and depth image as well as the pointcloud image.

The distance from the camera of the object is represented by the
center point of the boudning box.


@author: jschang
"""

# import sys model to appending paths
import sys
sys.path.append('D:\Jen\Projects\RealSense Camera\Codes\Python\selfbuildpackages')

# import self-build packages
from network_ssd import SSD
from PointCloudDrawer import AppState

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
ssd.load_model()
print("[INFO] network model loaded.")
# load the classes
CLASSES = ssd.load_classes()
# generate color list for the classes
COLORS = ssd.generate_colors(len(CLASSES))


## get the rs camera device
#ctx = rs.context()
#devices = ctx.query_devices()
#dev = devices[0]
#print("[INFO] Device name and iD {}:{}".format(dev.get_info(rs.camera_info.name), 
#        dev.get_info(rs.camera_info.product_id)))

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


# declare pointcloud instance
pc = rs.pointcloud()
# declare appState for pointcloud visualization
pc_state = AppState()
# declare a numpy array to store the pointcloud data for visualization
out = np.empty((480, 640, 3), dtype=np.uint8)
# declare colorizer instance
colorizer = rs.colorizer()

# initialize opencv visualization window
cv2.namedWindow("RGB video", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Pointcloud", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Pointcloud", pc_state.mouse_cb)



# start the rs-camera stream
profile = pipeline.start(config)
# get the camera intrinsics
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

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
        
        # map pointcloud to depth frame
        points = pc.calculate(depth_frame)
        pc.map_to(depth_frame)
        
        # point cloud data to numpy array
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asarray(v).view(np.float32).reshape(-1,3)        #xyz
        texcoords = np.asarray(t).view(np.float32).reshape(-1,2)    #uv
        
        
        
        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # depth image 3d. (depth image is 1-channel, color is 3 channels)
        depth_image_3d = np.dstack((depth_image*depth_scale,depth_image*depth_scale
                                    , depth_image*depth_scale))
        
        # Apply color map to depth image (image must converted to 8-bit per pixel first)
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
       
        
        
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
            color = ssd.get_color(idx)
            
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

##################
##  Visualization

        # render pointcloud
        out.fill(0)
        pc_state.grid(out, (0, 0.5, 1), size=1, n=10)
        pc_state.frustum(out, depth_intrinsics)
        pc_state.axes(out, pc_state.view([0, 0, 0]), pc_state.rotation, size=0.1, thickness=1)
        
        if not pc_state.scale or out.shape[:2] ==(h, w):
           pc_state.pointcloud_display(out, verts, texcoords, depth_colormap)
        
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pc_state.pointcloud_display(out, verts, texcoords, depth_colormap)
            tmp = cv2.resize(tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)        
        
        # monitoring mouse buttons
        if any(pc_state.mouse_btns):
            pc_state.axes(out, pc_state.view(pc_state.pivot), pc_state.rotation, thickness=4)
        
         
                
        # stack both image together
        images = np.hstack((color_image, depth_colormap_image))
        
        
        # show the image
#        cv2.namedWindow("Realsense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RGB video", images)
        cv2.imshow("Pointcloud", out)
        key = cv2.waitKey(1) & 0xFF
        
        # quit the loop by pressing 'q'
        if key == ord("q"):
            break
        
        
finally:
    pipeline.stop()
    cv2.destroyAllWindows()