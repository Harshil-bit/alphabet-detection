import cv2
from matplotlib import image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Setting an HTTPS context to fetch data from OpenML
if(not os.environ.get('PYTHONHTTPSVERIFIED','') and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context=ssl._create_unverified_context

#Fetching the data
X = np.load('image (3).npz')['arr_0']
y = pd.read_csv("P-122.csv")["labels"]
print(pd.Series(y).value_counts())
classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', "K", "L", "M",
"N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "Z"]
nclasses=len(classes)

#Splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)

#Scaling the features
X_train_scaled=X_train/255
X_test_scaled=X_test/255

#Fitting the training data into the model
clf=LogisticRegression(solver='saga', multi_class='multinomial').fit(X_test_scaled,y_train)

#Calculating the accuracy of the model
y_pred=clf.predict(X_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

#Starting the camera
cap=cv2.VideoCapture(0)

while(True):
    #Capturing frame by frame
    try:
        ret,frame=cap.read()

        #Converting BGR to gray
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Drawing the rectangular box at the centre of the video
        height, width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        #To only consider the area inside the box fro detecting the digit
        #ROI=region of interest
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        
        #Converting cv2 Image to PIL Image
        im_pil=Image.fromarray(roi)

        #Convert to gray-scale image- 'L' 
        #Format means each pixel is represented by a single value from 0 to 255
        image_bw=im_pil.convert('L')
        image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)

        #Inverting the image
        image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)

        pixel_filter=20
        #Converting to scalar quantity
        min_pixel=np.percentile(image_bw_resized_inverted,pixel_filter)
        
        #Using clip to limit the values between 0,255
        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel=np.max(image_bw_resized_inverted)

        #Converting into an array
        image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel

        #creating a text sample and predicting
        test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred=clf.predict(test_sample)
        print("Predicted class is ",test_pred)

        #Display the result in frame
        cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        pass

#When everything done release teh capture
cap.release()
cv2.destroyAllWindows()