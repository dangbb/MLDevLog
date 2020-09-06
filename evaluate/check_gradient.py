from linarg_package import *
import numpy as np

eps = 1e-6

def check_gradient(output_1, output_2, output_3):
    ### output_1: from evaluate function 1 with x as input
    ### output_2: from evaluate function 2 with x + delta as input
    ### output_3: derivative from formula

    ### return: print the differences

    print("Evaluate derivative function: ", abs((output_2 - output_1) / eps - output_3))

def cost(yhat, input):
    return np.sum(np.log(input))

def grad(yhat, input):
    return 1 / input

def binary_convterter(yhat):
    return (yhat > 0) * 1 + (yhat < 0) * 0

def normalize(y):
    cmin = y.min() - 1e-4
    cmax = y.max()

    y = (y - cmin) / (cmax - cmin + 1e-4)

    print("After normalize: Max: ", y.max(), " Min: ", y.min())
    # normalise so that data in range (0.1, 0.9)
    # make sure when + eps or - eps, the value doesn't exceed the range (0, 1)
    return y * 0.8 + 0.1

def check_Loss(num_experence):
    yhat = np.random.rand(10)

    for i in range(num_experence):
        y = np.random.rand(10)
        y = normalize(y)

        print(y)

        g = np.zeros(shape = 10)

        for j in range(10):
            y_u = y.copy()
            y_d = y.copy()
            y_u[j] += eps
            y_d[j] -= eps

            g[j] = (cost(yhat, y_u) - cost(yhat, y_d)) / (2 * eps) - grad(yhat, y)[j]

            if g[j] > 1e-6:
                print("Pos: ", j)
                print("Upper input: ", y_u)
                print("Lower_input: ", y_d)
                print("Upper_cost: ", cost(yhat, y_u))
                print("Lower_cost: ", cost(yhat, y_d))
                print("Grad: ", grad(yhat, y))

        print(g.max())

        if np.linalg.norm(g) < 1e-6:
            print("Safe")
        else:
            print("Not Safe")

check_Loss(1)





