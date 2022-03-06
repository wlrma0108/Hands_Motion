from cvzone.HandTrackingModule import HandDetector
import cv2

cap=cv2.VideoCapture(0)
while cap.isOpened():
  ret,mg=cap.read()
  if not ret:
    break

  cv2.imshow('cam',img)
  if cv2.waitKey(1)==ord('q'):
    break
  
  
  