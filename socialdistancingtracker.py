#importing opencv library for computer vision task

import cv2
from imutils.video import fps

#The Histogram of Oriented Gradients (HOG) descriptor is a method used in computer vision for object detection.
# It analyzes an image's gradient information, highlighting edges and texture patterns.
# By dividing the image into cells and creating histograms of gradient orientations, HOG captures local structure.


# creating instance of Hog descriptor for pedesterian detection.

hog = cv2.HOGDescriptor()

#Support Vector Machine (SVM) for detection using HOG descriptor is a machine learning approach.
# It employs a pre-trained SVM classifier to distinguish objects, like people, based on their HOG feature representations.
# SVMs find the optimal decision boundary, or hyperplane, to separate positive (object) and negative (non-object) examples, enabling accurate object detection.
# The `cv2.HOGDescriptor_getDefaultPeopleDetector()` retrieves a pre-trained SVM model specifically tailored for people detection,
# enhancing the effectiveness of the HOG-based object detection system.
# This combination of HOG features and SVM classification is a popular choice for object detection tasks in computer vision.
#https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Open a video file named 'video.mp4' for reading.

cap = cv2.VideoCapture('video.mp4')

#Initialize variables to keep track of the previous and current line lengths.

line_length_prev = 0
line_length_curr = 0

#Define the video codec and create a VideoWriter object to save the processed video as 'socialdistancing.avi'.

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('socialdistancing.avi', fourcc, 20.0, (1280, 720))

#Start an infinite loop for processing each frame of the video.

while True:
    #Read the next frame from the video.

    ret,img = cap.read()

    #Try to detect pedestrians using the HOG descriptor.

    try:
        boxes, weights = hog.detectMultiScale(img,winStride=(8,8))
    except:
        pass

    #If pedestrians are detected, iterate through the detected bounding boxes and draw rectangles around them and circles at their centers.

    else:
        for (x, y, w, h) in boxes:
            if w > 110:
                cv2.rectangle(img, (x, y), (x+w, y+h),(0, 255, 0), 2)
                cv2.circle(img,(x+int(w/2),y+int(h/2)),3,(0,0,255),3)

        #Check if there are at least two detected pedestrians.

        if len(boxes)>1:

            # Get the coordinates and dimensions of the first and second pedestrians.

            x1,y1,w1,h1 = boxes[0]
            x2,y2,w2,h2 = boxes[1]

            #Check if both pedestrians are sufficiently large (width > 110).

            if w1 > 110 and w2>110:

                #Calculate the current distance between the two pedestrians and draw a line connecting them. Euclidian distance

                line_length_prev = line_length_curr
                line_length_curr = int(((x1+(w1/2)-x2-(w2/2))**2 + (y1+(h1/2)-y2-(h2/2))**2)**0.5)
                if line_length_curr < line_length_prev/2:
                    continue
                cv2.line(img,(x1+int(w1/2),y1+int(h1/2)),(x2+int(w2/2),y2+int(h2/2)),(255,0,0),3)

                #Determine the minor and major object heights and display them on the image.

                min = (h2, h1) [h1 < h2]
                max = (h1, h2) [h1 < h2]
                cv2.putText(img,"current distance between them : "+str(line_length_curr),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                cv2.putText(img,'minor object height : '+str(min),(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                cv2.putText(img,'major object height : '+str(max),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

                #Check if the current distance between pedestrians is less than the height of the major object, indicating they are too close, and display a warning message.

                if line_length_curr < max:
                    cv2.putText(img,'too close!!',(600,300),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)


    #Finally, display the processed frame, write it to the output video, and check for a key press to exit the loop.

    finally:
        if ret==True:
            cv2.imshow('frame',img)
            out.write(img)
    if cv2.waitKey(1)==27:
        break
    # print(fps)

#Release the video capture and video writer objects, and close any OpenCV windows.
cap.release()
out.release()
cv2.destroyAllWindows()