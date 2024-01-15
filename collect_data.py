import os
import cv2

number_classes = 3
number_images = 100

data_dir = 'data'

cap = cv2.VideoCapture(0)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for j in range(number_classes):

    if not os.path.exists(os.path.join(data_dir,str(j))):
        os.makedirs(os.path.join(data_dir,str(j)))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame,'If Ready press ==> q',(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        cv2.imshow('re',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    print('Collecting data for Class ==> {}'.format(j))

    counter = 0

    while counter < number_images:
        ret,frame = cap.read()
        cv2.imshow('re',frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir,str(j),'{}.jpg'.format(counter)),frame)
        counter += 1
        
  
cap.release()
cv2.destroyAllWindows()    