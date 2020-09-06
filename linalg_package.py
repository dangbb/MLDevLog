import numpy as np
import h5py

eps = 0.0001

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis = 1, keepdims=True)

    return exp_x / sum_x

def image2vector(image):
    return image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

def normalizeRow(x):
    a = np.linalg.norm(x, axis = 1, keepdims=True)
    return x / a

def load_from(train_path):
    hf = h5py.File(train_path, 'r')
    keys = list(hf.keys())

    return np.array(hf.get(keys[1])), np.array(hf.get(keys[2]))

def load_dataset(train_path, test_path):
    return load_from(train_path), load_from(test_path)

def flatten_image(image):
    return image.reshape(-1, image.shape[0])

def L1(yhat, y):
    return np.sum(np.abs(yhat - y))

def L1_derivative(yhat, y):
    return ((yhat - y) >= 0) * (-1) + ((yhat - y) < 0) * 1

def L2(yhat, y):
    return 1/2 * np.dot(yhat - y, yhat - y)

def L2_derivative(yhat, y):
    return (y - yhat)

def cross_entropy(yhat, y):
    return -(np.dot(yhat, np.log(y).T) + np.dot(np.log(1-y), (1-yhat).T)) / y.size

def cross_entropy_derivative(yhat, y):
    return (- yhat / y + (1 - yhat) / (1 - y)) / y.size

