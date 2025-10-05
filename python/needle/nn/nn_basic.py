"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # Initialize weight with Kaiming Uniform initialization
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))

        # Initialize bias with Kaiming Uniform initialization (fan_in = out_features)
        if bias:
            bias_init = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = Parameter(bias_init.reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X is (N, in_features), weight is (in_features, out_features)
        # y = X @ weight^T + b = X @ weight + b (since weight is (in_features, out_features))
        out = X @ self.weight
        if self.bias is not None:
            # Broadcast bias explicitly
            out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X is (B, X_0, X_1, ...)
        # Flatten to (B, X_0 * X_1 * ...)
        batch_size = X.shape[0]
        feature_size = 1
        for i in range(1, len(X.shape)):
            feature_size *= X.shape[i]
        return X.reshape((batch_size, feature_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # softmax_loss(z, y) = log(sum(exp(z_i))) - z_y
        #                    = LogSumExp(z) - z_y
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        # Compute LogSumExp over axis 1 (classes)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))

        # Create one-hot encoding of y
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)

        # Extract z_y using one-hot encoding
        z_y = ops.summation(logits * y_one_hot, axes=(1,))

        # Compute loss: LogSumExp(z) - z_y
        loss = log_sum_exp - z_y

        # Return mean loss
        return ops.summation(loss) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # Initialize weight to 1 and bias to 0
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        # Running mean and variance for evaluation (not parameters)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x is (batch_size, dim)
        batch_size = x.shape[0]

        if self.training:
            # Compute batch statistics
            mean = ops.summation(x, axes=(0,)) / batch_size
            mean_broadcast = mean.reshape((1, self.dim)).broadcast_to(x.shape)

            centered = x - mean_broadcast
            variance = ops.summation(centered ** 2, axes=(0,)) / batch_size
            variance_broadcast = variance.reshape((1, self.dim)).broadcast_to(x.shape)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var.data + self.momentum * variance.data

            # Normalize
            normalized = centered / ops.power_scalar(variance_broadcast + self.eps, 0.5)
        else:
            # Use running statistics
            mean_broadcast = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            variance_broadcast = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)

            centered = x - mean_broadcast
            normalized = centered / ops.power_scalar(variance_broadcast + self.eps, 0.5)

        # Apply weight and bias
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)

        return weight * normalized + bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # Initialize weight to 1 and bias to 0
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x is (batch_size, dim)
        # Compute mean and variance over features (axis=1)
        batch_size = x.shape[0]

        # Mean over features
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = mean.reshape((batch_size, 1)).broadcast_to(x.shape)

        # Variance over features
        centered = x - mean
        variance = ops.summation(centered ** 2, axes=(1,)) / self.dim
        variance = variance.reshape((batch_size, 1)).broadcast_to(x.shape)

        # Normalize
        normalized = centered / ops.power_scalar(variance + self.eps, 0.5)

        # Apply weight and bias
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)

        return weight * normalized + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Create a mask using Bernoulli distribution
            # mask is 1 with probability (1-p), 0 with probability p
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            # Apply mask and scale by 1/(1-p)
            return x * mask / (1 - self.p)
        else:
            # During evaluation, just return the input (identity)
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
