from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # LogSoftmax(z) = z - LogSumExp(z)
        # Assuming 2D array and axis=1
        max_z = array_api.max(Z, axis=1, keepdims=True)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=1, keepdims=True)) + max_z
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # d/dz LogSoftmax(z) = I - softmax(z)
        # where softmax(z) = exp(log_softmax(z))
        z = node.inputs[0]
        # node is log_softmax(z), so softmax(z) = exp(node)
        softmax_z = exp(node)
        # Gradient: out_grad - sum(out_grad) * softmax(z)
        # sum over axis=1
        sum_out_grad = summation(out_grad, axes=(1,))
        # Reshape to (batch_size, 1)
        sum_out_grad_reshaped = sum_out_grad.reshape((sum_out_grad.shape[0], 1))
        # Broadcast to match shape
        sum_out_grad_broadcast = sum_out_grad_reshaped.broadcast_to(out_grad.shape)
        return out_grad - sum_out_grad_broadcast * softmax_z
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # LogSumExp(z) = log(sum(exp(z_i - max(z)))) + max(z)
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_reduce = array_api.max(Z, axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # d/dz LogSumExp(z) = exp(z - max(z)) / sum(exp(z - max(z)))
        #                    = exp(z - LogSumExp(z))
        z = node.inputs[0]

        # Get the shape for broadcasting
        shape = list(z.shape)
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(shape)))

        # Reshape node and out_grad to match z's shape
        new_shape = list(shape)
        for axis in axes:
            new_shape[axis] = 1

        node_reshaped = node.reshape(new_shape).broadcast_to(shape)
        out_grad_reshaped = out_grad.reshape(new_shape).broadcast_to(shape)

        return out_grad_reshaped * exp(z - node_reshaped)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)