import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dic = pickle.load(open('model.p','rb'))
model = model_dic['model']

mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils
mpdrawingstyle = mp.solutions.drawing_styles

hands = mphands.Hands(static_image_mode=True,min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)

output = {0:'A',1:'B',2:'C'}

while True:
    ret ,frame = cap.read()
    data_aux = []
    x_ = []
    y_ = []
    H,W,_ = frame.shape
    framergb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(framergb)
    if results.multi_hand_landmarks:
        for handlandmarks in results.multi_hand_landmarks:
            mpdrawing.draw_landmarks(
                frame,
                handlandmarks,
                mphands.HAND_CONNECTIONS,
                mpdrawingstyle.get_default_hand_landmarks_style(),
                mpdrawingstyle.get_default_hand_connections_style()
            )
            for i in range(len(handlandmarks.landmark)):
                x = handlandmarks.landmark[i].x
                y = handlandmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(handlandmarks.landmark)):
                x = handlandmarks.landmark[i].x
                y = handlandmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
        
        x1 = int(min(x_)*W) 
        y1 = int(min(y_)*H) 
        x2 = int(max(x_)*W)
        y2 = int(max(y_)*H)

        prediction =  model.predict([np.asarray(data_aux)])
        res = output[int(prediction[0])]
        # print(type(pred
        # iction[0]))
        # print(res)
        cv2.rectangle(frame,(x1-40,y1-30),(x2+30,y2+30),(0,0,0),5)
        cv2.putText(frame,res,(x1-40,y1-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)

    cv2.imshow('re',frame)
    if cv2.waitKey(10) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()