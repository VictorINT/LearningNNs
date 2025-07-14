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

def feedForward(PastLayer, LayerCount, cache=None):
    if cache is None:
        cache = {}
    if LayerCount > layer_count:
        return PastLayer, cache
    else:
        Z = W[LayerCount]["W"] @ PastLayer + W[LayerCount]["b"]
        if LayerCount == layer_count:
            A = softmax(Z)
        else:
            A = ReLU(Z)
        cache[LayerCount] = {"Z": Z, "A": A}
        return feedForward(A, LayerCount + 1, cache)


def calculateError(ExpectedOutput, ActualOutput):
    diff = ExpectedOutput - ActualOutput
    return np.sum(diff ** 2)

def backPropagation(x, y_true, cache, learning_rate):
    # x: input vector (shape: input_size x 1)
    # y_true: one-hot vector (10 x 1)
    # cache: dictionar cu Z si A pe fiecare layer
    # learning_rate: baza de invatare

    grads = {}
    m = 1  # batch size (1 pentru exemplul acesta)

    # 1) Calculam eroarea la output (layer 2)
    A2 = cache[2]["A"]  # output-ul softmax (10x1)
    dZ2 = A2 - y_true   # derivata functiei cost cross-entropy + softmax: simplifica la A - y_true

    # Gradientii pentru layer 2
    A1 = cache[1]["A"]  # activarea layerului 1 (128x1)
    dW2 = (dZ2 @ A1.T) / m  # (10x1)@(1x128) = 10x128
    db2 = dZ2 / m           # 10x1

    # 2) Calculam eroarea pentru layer 1
    W2 = W[2]["W"]         # 10x128
    dA1 = W2.T @ dZ2       # (128x10)@(10x1) = 128x1
    Z1 = cache[1]["Z"]
    dZ1 = dA1 * dReLU(Z1)  # element-wise multiplicare cu derivata ReLU (128x1)

    # Gradientii pentru layer 1
    x = x.reshape(-1,1)    # 784x1
    dW1 = (dZ1 @ x.T) / m  # (128x1)@(1x784) = 128x784
    db1 = dZ1 / m          # 128x1

    # 3) Adaptam learning rate proportional cu inversa normei gradientului (simplu)
    # Calculam norma totala a gradientilor
    grad_norm = np.linalg.norm(dW1) + np.linalg.norm(dW2)
    adaptive_lr = learning_rate / (grad_norm + 1e-8)  # +eps pentru stabilitate

    # 4) Actualizam weight-urile si bias-urile
    W[1]["W"] -= adaptive_lr * dW1
    W[1]["b"] -= adaptive_lr * db1
    W[2]["W"] -= adaptive_lr * dW2
    W[2]["b"] -= adaptive_lr * db2

for i in range(65000):
    x = X[i].reshape(-1, 1)
    y_true = np.zeros((10, 1))
    y_true[y[i]] = 1

    output, cache = feedForward(x, 1)
    err = calculateError(y_true, output)
    backPropagation(x, y_true, cache, learning_rate)

good_outcome = 0
bad_outcome = 0

for i in range(65000, len(X)):
    test = X[i].reshape(-1, 1)
    yt = np.zeros((10, 1))
    yt[y[i]] = 1

    out, _ = feedForward(test, 1)
    predicted_label = np.argmax(out)
    actual_label = y[i]

    if predicted_label == actual_label:
        good_outcome += 1
    else:
        bad_outcome += 1

accuracy = good_outcome / (good_outcome + bad_outcome) * 100
print(f"Accuracy: {accuracy:.2f}%")

