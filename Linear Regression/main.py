import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.utils import shuffle
import pickle

data = pd.read_csv('student_mat_2173a47420.csv',sep=';')
print(data.head())
# Attributes/ features pulled from the data

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
print(data.head())

# Label from the attributes based on the attributes this is the goal the prediction

predict = 'G3'
X = np.array(data.drop(predict, axis='columns'))
Y = np.array(data[predict])
# taking our attributes x_train is a selection of X  splitting 10% of the data into test samples so
# that when training does not have access to all data otherwise it would not be useful testing

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
'''
current_best = 0
for _ in range(30):
    # linear regression looks at a scatter of data points attempting to fit the line with a strength to the correlation  #
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)

    accuracy = linear.score(x_test,y_test)


    if accuracy > current_best:
        current_best = accuracy
        # saving the model
        print(accuracy)
        with open('Student_model.pickle','wb') as f:
            pickle.dump(linear,f)
'''
# Using the saved model
pickle_in = open('Student_model.pickle','rb')
linear = pickle.load(pickle_in)
print(linear.score(x_test,y_test))
# y = mx + c
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)
# indicates the predictions against the actual results based off of the training data

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

p = 'absences'
style.use('ggplot')
plt.scatter(data[p],data[predict])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()
