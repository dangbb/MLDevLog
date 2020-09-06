from linarg_package import *
import numpy as np

eps = 1e-8

def check_gradient(output_1, output_2, output_3):
    ### output_1: from evaluate function 1 with x as input
    ### output_2: from evaluate function 2 with x + delta as input
    ### output_3: derivative from formula

    ### return: print the differences

    print("Evaluate derivative function: ", abs((output_2 - output_1) / eps - output_3))

def cost(yhat, input):
    return L2(yhat, input)

def grad(yhat, input):
    return L2_derivative(yhat, input)

def check_L2(num_experence):
    yhat = np.random.normal(size = 10)

    for i in range(num_experence):
        y = np.random.normal(size = 10)
        g = np.zeros(shape = 10)

        for j in range(10):
            y_u = y.copy()
            y_d = y.copy()
            y_u[j] += eps
            y_d[j] -= eps

            g[j] = (cost(yhat, y_u) - cost(yhat, y_d)) / (2 * eps) - grad(yhat, y)[j]

        if np.linalg.norm(g) < 1e6:
            print("Safe")
        else:
            print("Not Safe")


check_L2(3)



