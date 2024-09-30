import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import operator
import collections
from keras.models import load_model
import time

import datetime
import WhatsAPPSender
from gtts import gTTS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def initTesting(adminnumber):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))#l1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))#l2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))#l3
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.load_weights('spitting_littering_model.h5')
    
    
        # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    
        # dictionary which assigns each label an emotion (alphabetical order)
    gesture_dict = {0:"blank",1:"littering", 2: "normal", 3: "spitting"}
    
        # start the webcam feed
    cap = cv2.VideoCapture(0)
    gesturelist=[]
   
    x=150
    y=100
    h=300
    w=300
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       # faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

      #  for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cv2.imwrite("temp.jpg",roi_gray)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
     
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        Gesturename=gesture_dict[maxindex]
        cv2.putText(frame, Gesturename, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
        gesturelist.append(Gesturename)
        frequency = collections.Counter(gesturelist)
        geturefreq=dict(frequency)
        sorted_d = sorted(geturefreq.items(), key=operator.itemgetter(1))
        print('Dictionary in ascending order by value : ',sorted_d)
        index=len(sorted_d)-1
        mxvaluegesture=sorted_d[index]
        print("Matched ",mxvaluegesture)
        finalname=mxvaluegesture[0]
        gesturecouunt=mxvaluegesture[1]
        print("finalname ",finalname)
        print("gesturecouunt ",gesturecouunt)
        count=int(gesturecouunt)
        if(count>=100):
            if(finalname=="littering"):
                gesturelist.clear()
                WhatsAPPSender.sendImage(adminnumber, "temp.jpg", "Littering Detected")
                print('Admin number '+adminnumber)
                language = 'en'
                voicetext=finalname+ "ALERT ALERT LITTERING IS DETECTED"
                myobj = gTTS(text=voicetext, lang=language, slow=False)
                myobj.save("say.mp3")
                os.system("say.mp3")
            if(finalname=="spitting"):
                gesturelist.clear()
                WhatsAPPSender.sendImage(adminnumber, "temp.jpg", "Spitting Detected")
                print('Admin number '+adminnumber)
                language = 'en'
                voicetext=finalname+ "ALERT ALERT SPITTING IS DETECTED"
                myobj = gTTS(text=voicetext, lang=language, slow=False)
                myobj.save("say.mp3")
                os.system("say.mp3")
     


              
        
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    initTesting()        