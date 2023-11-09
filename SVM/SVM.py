import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

##Classification dataset

cancer = datasets.load_breast_cancer()


##features/attributes and labels/targets
print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
Y = cancer.target
##random split of the data
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.2)

print(x_train, y_train)

classes = ['malignant','benign']

# used for classification or regression
# SVM (Support Vector Machine) creates a hyperplane (linearly) you can create a infinite amount of hyperplanes for a given set of data
# the best hyperplane to pick red data | green data has the most separation between the two groups
# To best seperate the data in the best way often for real data kernals are needed as the datacannot be seperated using a hyperline
# 2D to 3D using a kernal f(x2,x2) => x3

clf = svm.SVC(kernel='linear', C=2)
# comparing to KNN
#clf = KNeighborsClassifier(n_neighbors=9)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test,y_pred)
print(acc)