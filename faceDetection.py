import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("videos/3.mp4")
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDection = mpFaceDetection.FaceDetection(0.2)
pTime = 0

while True:
    success, img = cap.read()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDection.process(img)
    # print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            #print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxc = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin *
                                             ih), int(bboxc.width * iw), int(bboxc.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 2)
    cTime = time.time()
    if cTime != pTime:
        fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 200, 100), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(10)
