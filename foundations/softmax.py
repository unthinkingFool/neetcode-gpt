import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        arrNew=z-np.max(z)


        denom=0
        for i in range(z.shape[0]):
            denom=denom+np.exp(arrNew[i])
        
        arr = np.zeros(z.shape[0])

        for i in range(z.shape[0]):
            nom=np.exp(arrNew[i])
            arr[i]=nom/denom

        return np.round(arr, 4)
