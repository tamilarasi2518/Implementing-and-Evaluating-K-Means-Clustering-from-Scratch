import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        idx = np.argsort(dists)[:k]
        majority = Counter(y_train[idx]).most_common(1)[0][0]
        preds.append(majority)
    return np.array(preds)

if __name__ == '__main__':
    # synthetic dataset
    rng = np.random.default_rng(0)
    X0 = rng.normal([0,0], 1, (100,2))
    X1 = rng.normal([3,3], 1, (100,2))
    X = np.vstack([X0, X1])
    y = np.array([0]*100 + [1]*100)

    # shuffle and split
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = 150
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    preds = knn_predict(X_train, y_train, X_test, k=5)
    acc = (preds == y_test).mean()
    print("Accuracy:", acc)
