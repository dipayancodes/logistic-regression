import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(n_features):
    w = np.zeros((n_features, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = - (1 / m) * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8))
    dZ = A - Y
    dw = (1 / m) * np.dot(X, dZ.T)
    db = (1 / m) * np.sum(dZ)
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.6f}")
    params = {"w": w, "b": b}
    return params, grads, costs

def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

def model(X_train, Y_train, num_iterations=2000, learning_rate=0.01, print_cost=False):
    n_features = X_train.shape[0]
    w, b = initialize_parameters(n_features)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_train = predict(w, b, X_train)
    return params, Y_prediction_train, costs

if __name__ == '__main__':
    X = np.array([[1, 2, 3, 4],
                  [2, 3, 4, 5]])
    Y = np.array([[0, 0, 1, 1]])
    parameters, Y_prediction, costs = model(X, Y, num_iterations=1000, learning_rate=0.01, print_cost=True)
    print("Predictions on training set:", Y_prediction)
    print("Trained weights:", parameters["w"])
    print("Trained bias:", parameters["b"])
