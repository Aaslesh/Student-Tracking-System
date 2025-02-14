import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolo11n.pt')

area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('sample.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker=Tracker()
people_entering={}
entering=set()
people_exiting={}
exiting=set()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.xyxy  # Use xyxy instead of boxes

    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        if len(row) == 4:
            x1, y1, x2, y2 = map(int, row[:4])
            d = 0  # Assuming class information is not available in 4-value rows
        elif len(row) == 6:
            x1, y1, x2, y2, _, d = map(int, row)
        else:
            # Handle other cases if needed
            continue

        c = class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        ###people entering###
        results=cv2.pointPolygonTest(np.array(area2, np.int32),((x4,y4)),False)
        if results>=0:
                people_entering[id]=(x4,y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        if id in people_entering:
            results1=cv2.pointPolygonTest(np.array(area1, np.int32),((x4,y4)),False)
            if results1>=0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame,(x4,y4),5,(255,2,255),-1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                entering.add(id)
        ###people exiting###
        results2=cv2.pointPolygonTest(np.array(area1, np.int32),((x4,y4)),False)
        if results2>=0:
                people_exiting[id]=(x4,y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        if id in people_exiting:
            results3=cv2.pointPolygonTest(np.array(area2, np.int32),((x4,y4)),False)
            if results3>=0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                exiting.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    #print(people_entering)
    i=(len(entering))
    o=(len(exiting))
    cv2.putText(frame, str(i), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, str(o), (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
