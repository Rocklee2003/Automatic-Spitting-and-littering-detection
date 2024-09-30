
import cv2
import os

typename="blank"


datasetpath="DATASET"
if not os.path.exists(datasetpath):
    os.makedirs(datasetpath)         
        

typedataset="DATASET//"+typename
if not os.path.exists(typedataset):
    os.makedirs(typedataset) 
     
    
    
cap=cv2.VideoCapture(0)
k=0
x=150
y=100
h=300
w=300
while cap.isOpened():
    _,img=cap.read()
    cropimage = img[y:y+h, x:x+w]
    cropimage = cv2.cvtColor(cropimage, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    filename=str(k)   
    newfilepath=typedataset+"//"+filename+".jpg"
    dim = (48, 48)
    resized = cv2.resize(cropimage, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(newfilepath,resized)
    k=k+1   
            
          
    cv2.imshow('Capture Image( Press q to quit)',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
