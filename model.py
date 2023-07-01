# -*- coding: UTF-8 -*-

"""
模型类
"""

import numpy as np


class MyLinearWithLeakyRelu:
    def __init__(self, input_dim, output_dim):
        np.random.seed(0)
        self.W = np.random.normal(0, 2, (input_dim, output_dim))
        self.bias = np.random.normal(0, 2, (1, output_dim))
        self.leaky_relu_mask = None
        self.input_data = None
        self.new_W = None
        self.new_bias = None

    def leaky_relu(self, x):
        output = np.where(x >= 0, x, 0.1 * x)
        self.leaky_relu_mask = np.where(output >= 0, 1, 0.1)
        return output

    def forward(self, x):
        output = x @ self.W + self.bias
        self.input_data = x
        output = self.leaky_relu(output)
        return output

    def backward(self, next_grad, lr):
        a_grad = next_grad * self.leaky_relu_mask
        b_grad = a_grad
        W_grad = self.input_data.T @ a_grad

        self.new_bias = self.bias - lr * b_grad
        self.new_W = self.W - lr * W_grad

        return self.W @ a_grad.T

    def update(self):
        self.W = self.new_W
        self.bias = self.new_bias

        self.leaky_relu_mask = None
        self.input_data = None
        self.new_W = None
        self.new_bias = None


class MyDropout:
    def __init__(self, input_dim, drop_rate):
        self.input_dim = input_dim
        self.drop_rate = drop_rate
        self.drop_mask = None

    def forward(self, x):
        self.drop_mask = np.random.rand(*x.shape) > self.drop_rate
        return x * self.drop_mask

    def backward(self, next_grad, lr):
        return next_grad.T * self.drop_mask

    def update(self):
        self.drop_mask = None


class MyLinear:
    def __init__(self, input_dim, output_dim):
        np.random.seed(0)
        self.W = np.random.normal(0, 2, (input_dim, output_dim))
        self.bias = np.random.normal(0, 2, (1, output_dim))
        self.input_data = None
        self.new_W = None
        self.new_bias = None

    def forward(self, x):
        output = x @ self.W + self.bias
        self.input_data = x
        return output

    def backward(self, next_grad, lr):
        b_grad = next_grad
        W_grad = self.input_data.T @ next_grad

        self.new_bias = self.bias - lr * b_grad
        self.new_W = self.W - lr * W_grad

        return self.W @ next_grad.T

    def update(self):
        self.W = self.new_W
        self.bias = self.new_bias

        self.input_data = None
        self.new_W = None
        self.new_bias = None


