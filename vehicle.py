import cv2

import numpy as np


# to start web cam

cap = cv2.VideoCapture('tvid.mp4')


line_pos=450


# substructor alog

algo=cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


#to make a centre function
def center_point (x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy


count=[]
offset = 6 # error b/w pixels
vc=0

while True:
    ret, frame1 = cap.read()

    # h1,w1,_ =frame1.shape
    # # to customize the regin of intrest
    # roi=frame1[310:720,80:1280]

    # converting the image to gray scale 
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    ########_, gray = cv2.threshold(gray,244,245,cv2.THRESH_BINARY)


    # to convert into gaussin blur (reduce high definations)
    blur = cv2.GaussianBlur(gray,(3,3),5)

    #  to apply to all the frames
    img_sub = algo.apply(blur)

    dilat = cv2.dilate(img_sub,np.ones((5,5)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    d2 = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE, kernel)
    d2 = cv2.morphologyEx(d2,cv2.MORPH_CLOSE, kernel)

    counterShape , h = cv2.findContours(d2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # to draw the acceptance line according to the road/ req region

    cv2.line(frame1,(50,line_pos),(1300, line_pos),(255,127,0),3)


    # to draw a rectangular box for objs
    for(i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c) # extracting objects points

        # validation counter
        validation = (w >= 80 ) and (h>= 80)

        if(not validation):
            continue

        # drawing rectangle
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0,2))

        # to put a centre point
        center = center_point(x,y,w,h)
        count.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        # for the count
        for(x,y) in count:
            if(y< line_pos+offset) and y>(line_pos-offset):

               vc+=1

            cv2.line(frame1,(50,line_pos),(1300, line_pos),(0,127,255),3)
            count.remove((x,y))
           # print("vehicles count:"+ str(vc))


    cv2.putText(frame1,"Vehicle In-Count:"+str(vc),(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    cv2.imshow('video Original',frame1)

    if(cv2.waitKey(20) == 13):
        break

cv2.destroyAllWindows()
cap.release()