# kalman-pytorch
An implementation of the kalman filter with Pytorch.

The model support both linear and extended kalman fileters/smoothers. <br/>
It is possible to backpropagate through the filter to compute gradients with respect to the parameters and input.

In order to use the extended kalman, simply implement the System class with the correct linearize function.
