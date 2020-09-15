from linarg_package import *

import matplotlib.pyplot as plt

(train_x_orig, train_y), (test_x_orig, test_y) = load_dataset("train_catvnoncat.h5", "test_catvnoncat.h5")

image_size = train_x_orig.shape[1]
train_size = train_x_orig.shape[0]
test_size = test_x_orig.shape[0]

print("### Dataset examination ###")
print("Info: ", train_x_orig.shape)
print("Input image size: ", image_size)
print("Number of input in train set: ", train_size)
print("Number of input in test set: ", test_size)

print("Propotion in train: ", np.sum((train_y > 0.5)) / train_y.size * 100)
print("Propotion in test: ", np.sum((test_y > 0.5)) / test_y.size * 100)

train_y = train_y.astype(np.float64)
test_y = test_y.astype(np.float64)

##np.random.shuffle(train_x_orig)

print("### NORMALIZE DATA ###")
train_x = train_x_orig.reshape((train_x_orig.shape[0], -1)).T / 255
test_x = test_x_orig.reshape((test_x_orig.shape[0], -1)).T / 255

print("train set x shape: ", train_x.shape)
print("test set x shape: ", test_x.shape)
print("Input Examination: {} {} {} {}".format(train_x.max(), train_x.min(), test_x.max(), test_x.min()))

print()
print("### Training State ###")

param = fit(train_x, train_y, num_iteration=2000)

w = param['w']
b = param['b']

print("Param Examinate: {} {} {}".format(w.max(), w.min(), b))

train_predict_output = (sigmoid(np.dot(w.T, train_x) + b) > 0.5).astype(float)
train_accurate = 100 - np.mean(np.abs(train_predict_output - train_y) * 100)
test_predict_output = (sigmoid(np.dot(w.T, test_x) + b) > 0.5).astype(float)
test_accurate = 100 - np.mean(np.abs(test_predict_output - test_y) * 100)

print("Train Accuracy: {} Test Accuracy: {}".format(train_accurate, test_accurate))







