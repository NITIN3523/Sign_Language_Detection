import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils
mpdrawingstyle = mp.solutions.drawing_styles

hands = mphands.Hands(static_image_mode=True,min_detection_confidence=0.3)

data_dir = 'data'
data = []
labels = []

for dir in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir,dir)):
         
         data_aux = []
         x_ = []
         y_ = []

         imgpath = os.path.join(data_dir,dir,file)
         img = cv2.imread(imgpath)
         imgrgb = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)

         results = hands.process(imgrgb) # if image contain multiple hands all information store in it

         if results.multi_hand_landmarks: # enter if only any hand show in image
            for handlandmarks in results.multi_hand_landmarks: # if multi hands occurin image then handlandmarks in each iterate contanin information of one hand

                # mpdrawing.draw_landmarks(
                #     imgrgb,
                #     handlandmarks,
                #     mphands.HAND_CONNECTIONS,
                #     mpdrawingstyle.get_default_hand_landmarks_style(),
                #     mpdrawingstyle.get_default_hand_connections_style()
                # )
                
                for i in range(len(handlandmarks.landmark)): # handlandmarks.landmark contain location of all the points which show on each hand
                    x = handlandmarks.landmark[i].x
                    y = handlandmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(handlandmarks.landmark)): # handlandmarks.landmark contain location of all the points which show on each hand
                    x = handlandmarks.landmark[i].x
                    y = handlandmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir)

# print(len(data))
# print(len(labels))
# print(data_aux)
# for i in range(len(data)):
#     print(len(data[i]))

f = open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close

