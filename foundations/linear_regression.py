import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        predictions = np.dot( X , weights )
        return np.round(predictions,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        
        loss = model_prediction - ground_truth
        # the loss array is (n x 1) dim, so we have to flatten this array to get the 1D numpy array
        loss_1D = loss.flatten()

        mse = np.dot(
            loss_1D, loss_1D
        ) / model_prediction.shape[0]

        return np.round(mse,5)
