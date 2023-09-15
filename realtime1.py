import os
import cv2
from detecto import core, utils, visualize

from torchvision import datasets
import torchvision
import torch
import imutils
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import sendmailani as mail
Image.LOAD_TRUNCATED_IMAGES = True
model = core.Model.load('model_weights.pth', ['tiger', 'elephant', 'panda'])
def process1(image):
    
    predictions = model.predict(image)

    labels, boxes, scores = predictions

    scores=scores

    alt_score=[]
    for i in scores:
        alt_score.append(float(i))

    ele=[0]
    tig=[0]
    pan=[0]
    j=0
    for i in labels:
        if i=="elephant":
            ele.append(alt_score[j])
        elif i=="tiger":
            tig.append(alt_score[j])
        elif i=="panda":
            pan.append(alt_score[j])
        j=j+1
    final=[]    
    elephant_score=max(ele)
    tiger_score=max(tig)
    panda_score=max(pan)
    elephant_score=round(elephant_score*100,2)
    tiger_score=round(tiger_score*100,2)
    panda_score=round(panda_score*100,2)
    if (elephant_score>75):
        final.append("Elephant")
    else:
        final.append("None")
    if(tiger_score>75):
        final.append("Tiger")
    else:
        final.append("None")
    if(panda_score>75):
        final.append("Panda")
    else:
        final.append("None")
    print("Result==",final)
    prob=0.0
    if final[0]=="Elephant":
        prob=elephant_score
    if final[0]=="Tiger":
        prob=tiger_score
    if final[0]=="Panda":
        prob=panda_score
    if final[0]=="None":
        prob=0
    return final[0],prob
def process():
    camera = cv2.VideoCapture(0)
    
    while (True):
        (grabbed,frame) = camera.read()
        
        cv2.imwrite("atest.jpg",frame)
        #img=Image.open()
        image = utils.read_image("atest.jpg")
        pothole,prob = process1(image)
        if pothole=="Tiger" or pothole=="Panda" or pothole=="Elephant":
            mail.process("atest.jpg")
        clone = frame.copy()
        cv2.putText(clone , str(pothole)+' '+str(prob)+'%' , (30,30) , cv2.FONT_HERSHEY_DUPLEX , 1 , (0,255,0) , 1)

        #cv2.imshow("GrayClone",frame)

        cv2.imshow("Video Feed",clone)

        keypress = cv2.waitKey(1) & 0xFF

        if(keypress == ord("q")):
            break

    camera.release()

    cv2.destroyAllWindows()
#process()

        
        
