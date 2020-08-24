import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path, delim_whitespace=True, header=None)

    print("First five values: ")
    print(df.head())

    for name in df.keys():
        maxValue = df[name].max()
        df[name] = df[name] / maxValue

    return df.to_numpy()


def preprocess(dataset):
    (train, test) = train_test_split(dataset, test_size=0.2, random_state=42)
    (train, valid) = train_test_split(train, test_size=0.25, random_state=42)

    trainX = train[:, :13]
    trainY = train[:, 13:]

    validX = valid[:, :13]
    validY = valid[:, 13:]

    testX = test[:, :13]
    testY = test[:, 13:]

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
            self.bias = np.zeros(shape=(1, output_dim))
        else:
            self.bias = None

        self.weight = np.random.normal(size=(input_dim, output_dim)) * (np.sqrt(6) / np.sqrt(input_dim + output_dim))

        ### using Xavier initialization, weights are strictly in bound (-sqrt(6) / sqrt(in+out)) and (sqrt(6) / sqrt(in+out))

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
    def __init__(self, input_dim, model_path, verbose=True):
        self.input_dim = input_dim

        model_structure = pd.read_csv(model_path)

        self.depth = model_structure['input'].count()

        print("Network depth: ", self.depth)

        for i in range(self.depth):
            setattr(self, "layer_" + str(i), Layer(model_structure['input'][i], model_structure['output'][i]))
            print("New layer with input ", model_structure['input'][i], " and output ", model_structure['output'][i])

        self.predict = None
        self.verbose = verbose

        self.loss = lambda predict, groundTruth : 1/2 * np.mean(np.sum((predict - groundTruth) ** 2))


    def forward(self, input, update_grad=False):
        output = None
        for i in range(self.depth):
            if i == 0:
                output = getattr(self, "layer_" + str(i)).forward(input, update_grad)
            else:
                output = getattr(self, "layer_" + str(i)).forward(output, update_grad)
        if update_grad:
            self.predict = output

        return output


    def calc_loss(self, predict, groundTruth):
        return self.loss(predict, groundTruth)


    def backward(self, groundTruth):
        loss_grad = 1 / self.predict.shape[0] * (self.predict - groundTruth)

        for i in range(self.depth-1, -1, -1):
            getattr(self, "layer_" + str(i)).grad_compute(loss_grad)
            loss_grad = np.dot(loss_grad, getattr(self, "layer_" + str(i)).weight.T)

        for i in range(self.depth-1, -1, -1):
            getattr(self, "layer_" + str(i)).grad_update()

if __name__ == "__main__":
    net = NetWork(13, "model.csv")
    dataset = load_data("housing.csv")

    print("After preprocessing data sample: ")
    print(dataset[0])
    print()

    trainX, trainY, validX, validY, testX, testY = preprocess(dataset)

    batch_size = 16
    epochs = 120

    for epoch in range(epochs):
        loss = 0.0
        train_time = 0
        for index in range(0, trainX.shape[0], batch_size):
            border_right = min(trainX.shape[0], index + batch_size)
            data = trainX[index : border_right]

            train_predict = net.forward(data, True)
            loss += net.calc_loss(train_predict, trainY[index : border_right])

            net.backward(trainY[index : border_right])
            train_time += 1

        valid_predict = net.forward(validX, False)
        valid_loss = net.calc_loss(valid_predict, validY)
        print("Epoch : ", epoch, " Train Loss: ", loss / train_time, " Valid Loss: ", valid_loss)

    test_predict = net.forward(testX, False)
    test_loss = net.calc_loss(test_predict, testY)

    print("Test loss: ", test_loss)


