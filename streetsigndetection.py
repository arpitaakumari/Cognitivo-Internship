# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:34:37 2020

@author: Arpita Kumari
"""

import cv2
import numpy as np

#Load YOLO
print("Loading the model")
net = cv2.dnn.readNet("F:/Cognitivo/YOLOv4/YOLOv4-obj_last.weights","F:/Cognitivo/YOLOv4/YOLOv4-obj.cfg") 
classes = []
with open("F:/Cognitivo/YOLOv4/classname.txt","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("Loading the Image")    
print("Detection Started")
while True:
    frame = cv2.imread("F:/Cognitivo/YOLOv4/test_image_3.jpg")
    #frame = cv2.resize(frame, (720, 540))
    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320           
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    location=[]
    for out in outs:
        for detection in out:
            #start = timer()
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                #object detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                #rectangle co-ordinates
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes.append([x,y,w,h]) 
                confidences.append(float(confidence)) 
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]  
            center_x=(2*x+w)/2
            center_y=(2*y+h)/2
            label = str(classes[0])
            confidence= confidences[i]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255),2)
    #out.write(frame)
    cv2.imshow("StreetSign_Detection",frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    cv2.imwrite("StreetSign_deetction.jpg",frame)
print("Output image successfully saved")
print("Cleaning Up....")   
cv2.destroyAllWindows()