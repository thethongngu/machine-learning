from libsvm.svmutil import *
import numpy as np
from scipy.spatial import distance
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

# print("2.1")
# print("================Linear======================")
# model = svm_train(y_train, x_train, "-q -t 0")
# svm_predict(y_test, x_test, model)
#
# print(" ================Polynomial===================")
# model = svm_train(y_train, x_train, "-q -t 1")
# svm_predict(y_test, x_test, model)
#
# print(" ================RBF===================")
# model = svm_train(y_train, x_train, "-q -t 2")
# svm_predict(y_test, x_test, model)

# print("2.2")
# C = [0.1, 1, 10, 100]
# gamma = [0.25, 0.5, 1]
# coef0 = [0, 1, 10]
# degree = [1, 2, 4]
# best_acc = 0.0
# best_params = []
#
# for kernel_id in range(0, 3):
#     for cost_value in C:
#         if kernel_id == 0:  # linear
#             params = "-q -t 0 -v 10 " + "-c " + str(cost_value)
#             curr_acc = svm_train(y_train, x_train, params)
#             print(params)
#             if curr_acc > best_acc:
#                 best_acc = curr_acc
#                 best_params = [cost_value]
#
#         if kernel_id == 1:  # polynomial
#             for gamma_value in gamma:
#                 for coef0_value in coef0:
#                     for degree_value in degree:
#                         params = "-q -t 0 -v 10 " + \
#                                  "-c " + str(cost_value) + \
#                                  " -g " + str(gamma_value) + \
#                                  " -d " + str(degree_value) + \
#                                  " -r " + str(coef0_value)
#
#                         curr_acc = svm_train(y_train, x_train, params)
#                         print(params)
#                         if curr_acc > best_acc:
#                             best_acc = curr_acc
#                             best_params = [cost_value, gamma_value, degree_value, coef0_value]
#
#         if kernel_id == 2:  # RBF
#             for gamma_value in range(0, 3):
#                 params = "-q -t 0 -v 10 " + " -g " + str(gamma_value)
#                 curr_acc = svm_train(y_train, x_train, params)
#                 print(params)
#                 if curr_acc > best_acc:
#                     best_acc = curr_acc
#                     best_params = [cost_value, gamma_value]
#
# print("Best: " + str(best_acc))
# print("Params: " + str(best_params))

print("2.3")
gamma = 0.25
x_train_array = np.array(x_train)
kernel_train_linear = np.matmul(x_train, np.transpose(x_train))
kernel_train_RBF = distance.squareform(np.exp(-1 * gamma * distance.pdist(x_train_array, 'sqeuclidean')))
kernel_x_train = np.hstack((np.arange(1, 5001).reshape((5000, 1)), np.add(kernel_train_linear, kernel_train_RBF)))

x_test_array = np.array(x_test)
kernel_test_linear = np.matmul(x_test, np.transpose(x_train))  # (2500, 5000)
kernel_test_RBF = np.exp(-1 * gamma * distance.cdist(x_test_array, x_train_array, 'sqeuclidean'))
kernel_x_test = np.hstack((np.zeros((2500, 1)), np.add(kernel_test_linear, kernel_test_RBF)))

model = svm_train(y_train, kernel_x_train, "-q -t 4")
svm_predict(y_test, kernel_x_test, model)
