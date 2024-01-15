import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dic = pickle.load(open('data.pickle','rb'))

data = np.asarray(data_dic['data'])
labels = np.asarray(data_dic['labels'])

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

model = RandomForestClassifier()

# print(len(data))
# for i in range(len(data)):
#     print(len(data[i]))

model.fit(x_train,y_train)    

y_predit = model.predict(x_test)
score = accuracy_score(y_predit,y_test)

print("accuracy ==> ",str(score*100))

f = open('model.p','wb')
pickle.dump({'model' : model},f)
f.close