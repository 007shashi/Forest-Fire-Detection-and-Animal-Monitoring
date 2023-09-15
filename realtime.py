import os
import cv2
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
import sendmaila as mail
Image.LOAD_TRUNCATED_IMAGES = True
model = torch.load('./trained-models/model_final.pth',map_location=torch.device('cpu'))
class_names = class_names = ['Fire', 'Neutral', 'Smoke']
def predict(image):
   
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    #image = image.cuda()

    pred = model(image)
    idx = torch.argmax(pred)
    prob = pred[0][idx].item()*100
    
    return class_names[idx], prob
def process():
    camera = cv2.VideoCapture(0)
    
    while (True):
        (grabbed,frame) = camera.read()
        
        cv2.imwrite("test.jpg",frame)
        img=Image.open("test.jpg")
        pothole,prob = predict(img)
        if pothole=="Fire":
            mail.process("test.jpg")
            
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

        
        
