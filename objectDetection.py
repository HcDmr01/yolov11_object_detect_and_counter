import cv2
from ultralytics import YOLO
import tracker
import os

tracker = tracker.Tracker()

model = YOLO("Models/epoch_100_best.pt")
model.fuse()

cap = cv2.VideoCapture("Assets/falling_5.mp4")

total_count = 0

while True:
    _, frame = cap.read()
    # roi = frame[0:480,220:320]
    # cv2.rectangle(frame,(220,0),(320,480),(0,255,0),2)
    
    result = model(frame)

    
    xyxy=result[0].boxes.xyxy.cpu().numpy(),
    confidence=result[0].boxes.conf.cpu().numpy(),
    class_id=result[0].boxes.cls.cpu().numpy().astype(int)

    detections = []
    confidences = []
    class_ids = []

    if len(xyxy)>0:
        for obj in xyxy[0]:
            x1 = int(obj[0])
            y1 = int(obj[1])
            x2 = int(obj[2])
            y2 = int(obj[3])
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            detections.append([x1, y1, x2, y2])
        
        for conf in confidence[0]:
            confidences.append(round(conf,2))
        for id in class_id:
            class_ids.append(id)

    detected_obj = []
    
    if len(detections)>0:
        for i in range(0,len(detections)):
            if confidences[i] > 0.9:
                coords = detections[i]
                conf = confidences[i]
                id = class_ids[i]

                detected_obj.append(coords)
                #cv2.rectangle(frame,(coords[0],coords[1]),(coords[2],coords[3]),(0,0,255),2)
                #cv2.putText(frame, str(conf), (coords[0],coords[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    obj_ids = tracker.update(objects_rect=detected_obj)
    for obj_id in obj_ids:
        x1, y1, x2, y2, id = obj_id
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        text = f"#{id}"
        cv2.putText(frame, text, (coords[0],coords[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if id>total_count:
            total_count = id
    text = f"Total Number of Objects: {total_count}"
    cv2.putText(frame, text, (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    cv2.imshow("frame", frame)
    # cv2.imshow("roi", roi)

    if cv2.waitKey(15) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()