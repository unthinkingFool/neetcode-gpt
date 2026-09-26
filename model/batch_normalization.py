import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists]

        esp=1e-5
        x = np.array(x, dtype = float)
        gamma = np.array(gamma , dtype = float)
        beta = np.array( beta , dtype = float )

        running_mean = np.array( running_mean, dtype = float )
        running_var = np.array(running_var , dtype = float )

        if training : 
            # x => (batch_size, features)
            num_features=x.shape[1]
            mean=[]
            var=[]
            for i in range(num_features):

                mean_i = np.mean(x[:,i])
                mean.append(mean_i) 

                var_i = np.mean((x[:,i]-mean_i)**2)
                var.append(var_i)

            mean = np.array(mean)
            var=np.array(var)

            x_hat = ( x - mean ) / (( var + eps )**0.5)

            # updating the running mean and running var
            #(1-momentum)*running_mean + momentum*mean
            #(1-momentum)*running_var + momentum*var
            m=momentum
            running_mean = (1-m)*running_mean + m * mean
            running_var = (1-m)*running_var + m*var


        else :

            x_hat = ( x - running_mean ) / (( running_var + esp )**0.5)

        y = gamma * x_hat + beta

        return (
            np.round(y,4).tolist(),
            np.round(running_mean,4).tolist(),
            np.round(running_var,4).tolist()
        )


        
