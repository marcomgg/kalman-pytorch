# kalman-pytorch
An implementation of the kalman filter with Pytorch.

The model supports both linear and extended kalman fileters/smoothers. <br/>
It is possible to backpropagate through the filter to compute gradients with respect to the parameters and input.

In order to use the extended kalman filter, simply derive the NonLinearSystem class and implement <br/>
the f and g members using pytorch functions.
