import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        num_features=X.shape[1]
        num_samples=X.shape[0]
        b=0
        w=np.zeros(num_features)

        for i in range(epochs):
            y_hat=X @ w + b
            loss=np.mean((y_hat-y)**2)

            dl_dw = ( 2*(y_hat-y) @ X)/num_samples
            dl_db =  2 * np.sum(y_hat-y) / num_samples

            w=w-lr*dl_dw
            b=b-lr*dl_db

        return np.round(w,5), np.round(b,5)
