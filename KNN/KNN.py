
import numpy as np
import pandas as pd

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv('car.data')
print(data.head())


### preprocessing into integer values  using sklearn

## new object##

le = preprocessing.LabelEncoder()
##attributes list ##

buying = le.fit_transform(list(data['buying'])) # this tales the buying column and converts into a list
# transform that list into appropiate integer values
maint = le.fit_transform(list(data['maint']))
door= le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

# turns in an array
print(buying, type(buying))
#features
X = list(zip(buying,maint,door,persons,lug_boot,safety))

# labels
Y = list(cls)


x_train,x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
print(x_train,y_train)

## creating classifier
num_neighbors = 7
model = KNeighborsClassifier(n_neighbors=num_neighbors)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)

print(acc)
predicted = model.predict(x_test)
names = ['unacc','acc','good','vgood']

for x in range(len(predicted)):
    print('Predicted Data: ', names[predicted[x]],'Data: ', x_test[x],'Actual: ', names[y_test[x]])
    n = model.kneighbors([x_test[x]],num_neighbors,True)
    print('N: ',n)


