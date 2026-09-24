import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        bce = -(
            np.dot(
                y_true,
                np.log(y_pred_clipped)
            )
            +
            np.dot(
                (1 - y_true),
                np.log(1 - y_pred_clipped)
            )
        ) / y_true.shape[0]

        return round(bce, 4)


    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        cce = 0

        for true_sample, pred_sample in zip(y_true, y_pred_clipped):

            temp_cce = -np.dot(
                true_sample,
                np.log(pred_sample)
            )

            cce += temp_cce

        cce = cce / y_true.shape[0]

        return np.round(cce, 4)