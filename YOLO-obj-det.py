import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

yolo = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
# classes = []

with open("coco.names", 'r') as f:
    classes = f.read().splitlines()

# print(classes)
width = 320
height = 320

img = cv2.imread("test11.jpg")
img = imutils.resize(img, width=width, height=height)
blob = cv2.dnn.blobFromImage(img, 1/255, (width, height), (0, 0, 0), swapRB = True, crop = True)
# print(blob)
i = blob[0].reshape(width,height,3)
cv2.imshow('blob',i)
cv2.waitKey(0)

yolo.setInput(blob)
output_layer_names = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layer_names)

boxes = []
confidences = []
class_ids = []

for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.5:
            centre_x = int(detection[0]*width)
            centre_y = int(detection[1]*height)
            print(centre_x, centre_y)
            w = int(detection[2]*width)
            h = int(detection[3]*height) + 50

            x = int(centre_x - w/4)
            # x = int(centre_x)
            y = int(centre_y - h/2)
            # y = int(centre_y)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)



font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size = (len(boxes),3))

# for i in range(len(boxes)):
#     x,y,w,h = boxes[i]
#     label = str(classes[class_ids[i]])
#     confi = str(round(confidences[i], 2))
#     color = colors[i]
#
#     cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#     cv2.putText(img, label +" " + confi, (x,y), font, 2, (255,255,255), 2)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(len(boxes))

# font = cv2.FONT_HERSHEY_PLAIN
# colors = np.random.uniform(0, 255, size = (len(boxes),3))

for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i], 2))
    color = colors[i]

    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label +" " + confi, (x,y), font, 2, (255,255,255), 2)
cv2.imshow('output',img)
cv2.waitKey(0)

