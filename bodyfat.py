import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    for name in df.keys():
        maxValue = df[name].max()
        df[name] = df[name] / maxValue

    return df.to_numpy()


def preprocess(dataset):
    (train, test) = train_test_split(dataset, test_size=0.2, random_state=42)
    (train, valid) = train_test_split(train, test_size=0.25, random_state=42)

    trainX = train[:, 1:]
    trainY = train[:, :1]

    validX = valid[:, 1:]
    validY = valid[:, :1]

    testX = test[:, 1:]
    testY = test[:, :1]

    return trainX, trainY, validX, validY, testX, testY


class A_Layer():
    def __init__(self, type="relu"):
        self.type = type
        self.input = None

        if type == "relu":
            self.layer = lambda input : np.maximum(input, np.zeros_like(input))
            self.grad = lambda output : (output > 0).astype(int)
        elif type == "identity":
            self.layer = lambda input : input
            self.grad = lambda input : np.ones_like(input)
        else:
            raise Exception("Activation Layer type " + type + " not available")


    def forward(self, input, update_grad=True):
        if update_grad:
            self.input = input
        return self.layer(input)


    def grad_compute(self):
        return self.grad(self.input)


class Layer():
    def __init__(self, input_dim, output_dim, bias=True, learning_rate=0.001, a_layer="relu"):
        print("Layer Weight size: ", input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.haveBias = bias
        self.learning_rate = learning_rate

        self.a_layer = A_Layer(a_layer)

        if bias:
            self.bias = np.random.normal(size=(1, output_dim))
        else:
            self.bias = None

        self.weight = np.random.normal(size=(input_dim, output_dim))
        self.input = None
        self.grad_weight = None
        self.grad_bias = None


    def forward(self, input, update_grad=False):
        if update_grad:
            self.input = input
        output = np.dot(input, self.weight)

        if self.haveBias:
            output = output + self.bias

        output = self.a_layer.forward(output, update_grad)
        return output


    def grad_compute(self, output_grad):
        output_grad = self.a_layer.grad_compute() * output_grad
        self.grad_weight = np.dot(self.input.T, output_grad)

        if self.haveBias:
            self.grad_bias = np.sum(output_grad, axis=0, keepdims=True)

        return output_grad


    def grad_update(self):
        self.weight = self.weight - self.learning_rate * self.grad_weight

        if self.haveBias:
            self.bias = self.bias - self.learning_rate * self.grad_bias
        self.grad_bias = None
        self.grad_weight = None


class NetWork():
    def __init__(self, input_dim, verbose=True):
        self.input_dim = input_dim
        self.layer_1 = Layer(input_dim, 15, True)
        self.layer_2 = Layer(15, 8, True)
        self.layer_3 = Layer(8, 1, True)

        self.predict = None
        self.verbose = verbose

        self.loss = lambda predict, groundTruth : 1/2 * np.mean(np.sum((predict - groundTruth) ** 2))


    def forward(self, input, update_grad=False):
        output = self.layer_1.forward(input, update_grad)
        output = self.layer_2.forward(output, update_grad)
        output = self.layer_3.forward(output, update_grad)
        if update_grad:
            self.predict = output
        return output


    def calc_loss(self, predict, groundTruth):
        return self.loss(predict, groundTruth)


    def backward(self, groundTruth):
        self.calc_loss(self.predict, groundTruth)
        loss_grad = 1 / self.predict.shape[0] * (self.predict - groundTruth)

        self.layer_3.grad_compute(loss_grad)
        loss_grad = np.dot(loss_grad, self.layer_3.weight.T)
        self.layer_2.grad_compute(loss_grad)
        loss_grad = np.dot(loss_grad, self.layer_2.weight.T)
        self.layer_1.grad_compute(loss_grad)

        self.layer_3.grad_update()
        self.layer_2.grad_update()
        self.layer_1.grad_update()



data_dim = 14

dataset = load_data("body.csv")
trainX, trainY, validX, validY, testX, testY = preprocess(dataset)

net = NetWork(13)

epochs = 150

for epoch in range(epochs):
    train_predict = net.forward(trainX, True)
    valid_predict = net.forward(validX, False)
    if epoch % 5 == 0:
        print("Epoch: ", epoch, end = "")
        print(" Train Loss: ", net.calc_loss(train_predict, trainY), " Valid Loss: ", net.calc_loss(valid_predict, validY))

    net.backward(trainY)
    break

test_predict = net.forward(testX, False)
print(" Test Loss: ", net.calc_loss(test_predict, testY))

