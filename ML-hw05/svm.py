from libsvm import svmutil

import csv

# Read data
with open('X_train.csv') as f:
    reader = csv.reader(f, delimiter=',')
    x_train = [[float(y) for y in x] for x in list(reader)]

with open('Y_train.csv') as f:
    reader = csv.reader(f, delimiter=',')
    y_train_list = [[float(y) for y in x] for x in list(reader)]
    y_train = [y for y_list in y_train_list for y in y_list]

with open('X_test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    x_test = [[float(y) for y in x] for x in list(reader)]

with open('Y_test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    y_test_list = [[float(y) for y in x] for x in list(reader)]
    y_test = [y for y_list in y_test_list for y in y_list]

# 2.1
print("================Linear======================")
model = svmutil.svm_train(y_train, x_train, "-t 0")
svmutil.svm_predict(y_test, x_test, model)

print(" ================Polynomial===================")
model = svmutil.svm_train(y_train, x_train, "-t 1")
svmutil.svm_predict(y_test, x_test, model)

print(" ================RBF===================")
model = svmutil.svm_train(y_train, x_train, "-t 2")
svmutil.svm_predict(y_test, x_test, model)

