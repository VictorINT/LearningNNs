import numpy as np
from sklearn.datasets import fetch_openml

input_size = 28*28
hidden_size = 128
output_size = 10
learning_rate = 0.1

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.array(X) / 255.0
y = np.array(y).astype(int)

layer_count = 2

W = {
    1: {
        "W": np.random.randn(hidden_size, input_size) * 0.01,
        "b": np.zeros((hidden_size, 1))
    },
    2: {
        "W": np.random.randn(output_size, hidden_size) * 0.01,
        "b": np.zeros((output_size, 1))
    }
}

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return x > 0

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def feetForward(PastLayer, LayerCount):
    if LayerCount > layer_count:
        return PastLayer
    else:
        Z = W[LayerCount]["W"] @ PastLayer + W[LayerCount]["b"]
        if LayerCount == layer_count:
            A = softmax(Z)
        else:
            A = ReLU(Z)
        return feetForward(A, LayerCount + 1)

def calculateError(ExpectedOutput, ActualOutput):
    diff = ExpectedOutput - ActualOutput
    return np.sum(diff ** 2)

def main():
    x0 = X[0].reshape(-1, 1)
    y0 = np.zeros((10, 1))
    y0[y[0]] = 1

    outcome = feetForward(x0, 1)
    err = calculateError(y0, outcome)

    print("Predictie (distributie probabilitati):", outcome.flatten())
    print("Eroare fata de realitate:", err)

main()
