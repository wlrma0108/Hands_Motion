import cv2
import mediapipe as mp
import numpy as np



mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

hands=mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

file=np.genfromtxt('gesture_train.csv',delimiter=',')
angle= file[:,:-1].astype(np.float32)
label=file[:,-1].astype(np.float32)

knn=cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE,label)

cap=cv2.VideoCapture(0)


while cap.isOpened():
    
    ret,img=cap.read()
    
    if not ret:
        break
    
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.flip(img,1)
    
    result=hands.process(img)
    
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j,lm in enumerate(res.landmark):
                joint[j]=[lm.x,lm.y,lm.z]
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v=v2-v1
            
            v=v/np.expand_dims(np.linalg.norm(v,axis=1),axis=-1)
            
            angle=np.arccos(np.einsum('nt,nt->n',
                                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle=np.degrees(angle)
            angle=np.expand_dims(angle.astype(np.float32),axis=0)

            ret, results, neighbours, dist = knn.findNearest(angle, 3)
            print(results)
            
            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
    cv2.imshow('result',img)
    if cv2.waitKey(1)==ord('q'):
        break