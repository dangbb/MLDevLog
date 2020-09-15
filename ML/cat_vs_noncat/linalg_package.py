import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import h5py

def load_from(train_path):
    hf = h5py.File(train_path, 'r')
    keys = list(hf.keys())

    return np.array(hf.get(keys[1])), np.array(hf.get(keys[2]))

def load_dataset(train_path, test_path):
    return load_from(train_path), load_from(test_path)

def flatten(x):
    s = x.reshape(-1, x.shape[0])
    return s

def binary_cross_entropy(yhat, y):
    m = y.size
    return - (np.dot(yhat, np.log(y).T) + np.dot((1 - yhat), np.log(1 - y).T)) / m

def binary_cross_entropy_derivative(yhat, y):
    return (- yhat / y + (1 - yhat) / (1 - y)) / y.size

def initialize(size, type = 'zero'):
    w = None
    b = 0

    if type == 'zero':
        w = np.zeros((size, 1))
    elif type == 'normal':
        w = np.random.normal(size = (size, 1))
    elif type == 'xavier':
        w = np.random.normal(size = (size, 1)) * 6 / np.sqrt(size + 1)

    assert (w.shape == (size, 1))
    assert (isinstance(b, int) or isinstance(b, float))
    return w, b

def propagation(w, b, X, Y):
    raw_output = np.dot(w.T, X) + b
    m = Y.size
    act_output = sigmoid(raw_output)

    cost = - (np.dot(Y, np.log(act_output).T) + np.dot(np.log(1 - act_output), (1 - Y).T)) / m

    dw = np.dot(X, (act_output - Y).T) / m
    db = np.sum(act_output - Y) / m

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost, act_output, raw_output

def predict(w, b, X):
    a = sigmoid(np.dot(w.T, X) + b)

    return (a > 0.5)


def optimize(X, Y, num_iterations = 200, learning_rate = 0.005):
    w, b = initialize(X.shape[0])
    costs = []
    accuracies = []

    for i in range(num_iterations):
        grads, cost, act_output, raw_output = propagation(w, b, X, Y)
        label = (act_output > 0.5)
        accuracy = np.mean(np.abs(label - Y) * 100)

        dw = grads["dw"]
        db = grads["db"]

        print("Epoch: ", i)
        print("   Examinate: ")
        print("   Grad: ", dw.max(), dw.min(), db)
        print("   Cost: ", cost)
        print("   Accuracy: ", accuracy)
        print("   Act Output: ", act_output.max(), act_output.min())
        print("   Raw Output: ", raw_output.max(), raw_output.min())
        print("   weight: ", w.max(), w.min())
        print("   bias: ", b)

        costs.append(cost)
        accuracies.append(accuracy)
        w = w - learning_rate * dw
        b = b - learning_rate * db

    parameter = {
        "w": w,
        "b": b
    }

    return parameter, costs, accuracies

def model(train_x, train_y, test_x, test_y):
    param, costs, accuracies = optimize(train_x, train_y)

    w = param['w']
    b = param['b']
    train_predict = predict(w, b, train_x)
    test_predict = predict(w, b, test_x)

    train_predict_accurate = np.mean(np.abs(train_predict - train_y) * 100)
    test_predict_accurate = np.mean(np.abs(test_predict - test_y) * 100)

    print()
    print("Result: ")
    print("Train accurate: ", train_predict_accurate)
    print("Test accurate: ", test_predict_accurate)

    return param

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def fit(train_x, train_y, w = None, b = None, num_iteration=0, learning_rate=0.005, debug=False):

    if w is None:
        w = np.zeros(shape = (train_x.shape[0], 1))
    if b is None:
        b = 0

    assert (w.shape == (train_x.shape[0], 1))
    assert (isinstance(b, int) or isinstance(b, float))

    costs = []

    for i in range(num_iteration):

        output = np.dot(w.T, train_x) + b
        A = sigmoid(output)

        m = train_x.shape[1]

        if debug:
            print("........... Output Examinate: {} {} {}".format(output.min(), output.max(), m))

        cost = -(np.dot(train_y, np.log(A).T) + np.dot(np.log(1 - A), (1 - train_y).T)) / m
        dw = np.dot(train_x, (A - train_y).T) / m
        db = np.sum(A - train_y) / m

        cost = np.squeeze(cost)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if debug:
            print("........ Grad Examinate: {} {} {}".format(dw.max(), dw.min(), db))

        if i % 100 == 0:
            costs.append(cost)
            print("Iteration: {} Cost: {}".format(i, cost))

    parameter = {
        "w":w,
        "b":b
    }
    return parameter

def testing():
    w = np.array([[1.], [2.]])
    b = 2.
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])
    param = fit(X, Y, w, b, num_iteration=100, learning_rate=0.009)
    print("w = ", param['w'])
    print("b = ", param['b'])















        

        





    










