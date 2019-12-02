# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:48:25 2019

Objective:
    Use Realsense RGB-D camera with real-time object detection model




@author: jschang
"""

# import necessary packages
import numpy as np
import pyrealsense2 as rs
import cv2
import time
from imutils.video import FPS


# configure the realsense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640 ,480, rs.format.bgr8, 15)
print("[INFO] rs camera configured.")

# Setting for object detection model (caffe)
dir_coffe_prototxt = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\real-time-object-detection\\MobileNetSSD_deploy.prototxt.txt"
dir_coffe_model = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\real-time-object-detection\\MobileNetSSD_deploy.caffemodel"
confidence_thresh = 0.5


# initialize the list of class labels MobileNet SSD was trained to detect
# and generate a set of bonding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.randint(0, 255, size = (len(CLASSES), 3), dtype="uint8")


# load the pre-trained network
net = cv2.dnn.readNetFromCaffe(dir_coffe_prototxt, dir_coffe_model)
print("[INFO] network model loaded.")


# start the rs-camera stream
pipeline.start(config)
time.sleep(2.0)
fps = FPS().start()

# loop over frames from the stream
try:
    while True:
        
        # wait for a coherent pair of frames: color and depth frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        
        # apply color map to the depth image
        # image must converted to 8-bit per pixel first
        depth_colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                               alpha=0.03), cv2.COLORMAP_JET)
        
        
        
        # grab the frame dimensions and convert it to a blob
        (h, w) = color_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 
                                     0.007843, (300, 300), 127, 5)
        # pass the blob thought the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()
        
    
        
        
        #loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence 
            confidence = detections[0, 0, i, 2]
            
            # filter out the weak detections
            if confidence > confidence_thresh:
                # extract the class index
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                
                # find the bounding box
                box = detections[0, 0, i, 3:7] * np.asarray([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                
                # draw the bounding box and text
                color = [int(c) for c in COLORS[idx]]
                text = "{}: {:.2f}%".format(label, confidence * 100)
                cv2.rectangle(color_image, (startX, startY), 
                                  (endX, endY), color, 2)
                text_y = startY-15 if startY - 15 > 15 else startY + 15
                cv2.putText(color_image, text, (startX, text_y), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                
       
        # stack both image together
        images = np.hstack((color_image, depth_colormap_image))
        
        # show the image
        cv2.namedWindow("Realsense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Realsense", images)
        key = cv2.waitKey(1) & 0xFF
        
        
        # update FPS count
        fps.update()
        
        # quit the loop by pressing 'q'
        if key == ord("q"):
            break
        

finally:
    
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time:{:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS:{:.2f}".format(fps.fps()))
    
    # stop streaming and clean up windows
    pipeline.stop()
    cv2.destroyAllWindows()
    
    
    
    









