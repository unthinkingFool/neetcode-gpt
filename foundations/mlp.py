import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        num_layer=len(weights)
        
        ans=x

        for i in range(num_layer):

            y_hat= np.dot(ans,weights[i])+biases[i]

            if i<num_layer-1:
                ans=np.maximum(y_hat,0)
            else:
                ans=y_hat

        return np.round(ans,5)
