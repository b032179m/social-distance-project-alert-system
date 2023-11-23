import cv2
from imutils.video import FPS
import matplotlib.pyplot as plt
from collections import deque

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('video.mp4')

line_length_prev = 0
line_length_curr = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('socialdistancing.avi', fourcc, 20.0, (1280, 720))

# Initialize the FPS counter
fps = FPS().start()
fps_values = deque(maxlen=200)  # Keep the last 200 FPS values for plotting

while True:
    ret, img = cap.read()

    try:
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
    except:
        pass
    else:
        for (x, y, w, h) in boxes:
            if w > 110:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(img, (x+int(w/2), y+int(h/2)), 3, (0, 0, 255), 3)

        if len(boxes) > 1:
            x1, y1, w1, h1 = boxes[0]
            x2, y2, w2, h2 = boxes[1]

            if w1 > 110 and w2 > 110:
                line_length_prev = line_length_curr
                line_length_curr = int(((x1+(w1/2)-x2-(w2/2))**2 + (y1+(h1/2)-y2-(h2/2))**2)**0.5)
                if line_length_curr < line_length_prev/2:
                    continue
                cv2.line(img, (x1+int(w1/2), y1+int(h1/2)), (x2+int(w2/2), y2+int(h2/2)), (255, 0, 0), 3)

                min_h = min(h2, h1)
                max_h = max(h1, h2)
                cv2.putText(img, "current distance between them: "+str(line_length_curr), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(img, 'minor object height: '+str(min_h), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(img, 'major object height: '+str(max_h), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if line_length_curr < max_h:
                    cv2.putText(img, 'too close!!', (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Calculate and display fps
    fps.update()
    fps.stop()
    current_fps = fps.fps()
    fps_values.append(current_fps)

    # Plot the FPS graph
    plt.clf()  # Clear the previous plot
    plt.plot(fps_values, label='FPS')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.title('Frames Per Second Over Time')
    plt.legend()
    plt.pause(0.01)  # Pause to allow the plot to update

    cv2.putText(img, "FPS: {:.2f}".format(current_fps), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if ret == True:
        cv2.imshow('frame', img)
        out.write(img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
