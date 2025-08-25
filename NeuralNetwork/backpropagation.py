"""
Autograd Engine (Mini-Grad)
===========================

This is a step-by-step implementation of the gradient descent algorithm
modeled after PyTorch's `autograd` functionality. The goal is to show
how automatic differentiation works under the hood by building a minimal
computational graph system.

Features
--------
- `Value` objects store data and track computation history
- Support for backward pass via reverse-mode automatic differentiation
"""

import math


class Value:
    """
    A scalar value in a computational graph.

    This class stores a data value and maintains references to its
    predecessors in the graph, enabling automatic differentiation
    through backpropagation.

    Attributes
    ----------
    data : float
            The numeric value of the node.
    grad : float
            The gradient of some loss with respect to this value.
    _prev : set
            Set of parent `Value` objects that produced this node.
    _backward : callable
            Function to propagate gradients to parent nodes.
    """

    def __init__(self, data: float, _children=()):
        """
        Initialize a new `Value` node.

        Parameters
        ----------
        data : float
                The numeric value stored in this node.
        _children : iterable of Value, optional
                The parent nodes that created this node (default is empty).
        grad: float
                Store the gradient value
        """

        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        """
        Return a string representation of the Value object.
        Returns
        -------
        str
        A string showing the value and gradient.
        """
        return f"value(data= {self.data}), grad={self.grad}"

    def __add__(self, other):
        assert isinstance(other, Value), ("Added object should be "
                                          "of class Value")
        out = Value(self.data+other.data, (self, other))

        def _backward():

            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        assert isinstance(other, Value), ("Multiplied object should be"
                                          " of class Value")
        out = Value(self.data*other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        assert isinstance(other, Value), ("Subtracted object should be of"
                                          " class Value")
        out = Value(self.data - other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        assert isinstance(other, Value), ("Subtracted object should be of "
                                          "class Value")
        try:
            out = Value(self.data / other.data, (self, other))
        except ZeroDivisionError:
            print(f"Value of divisor is negative: {other.data}")
        return out

    def __pow__(self, power, modulo=None):
        assert isinstance(power, (int, float)), ("Only integer and floating"
                                                 " points for powers")
        out = Value(self.data**power, (self,))

        def _backward():
            self.grad += out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __ln__(self):
        assert self.data > 0, "Log is undefined for non-positive values"
        out = Value(math.log(self.data), (self,))

        def _backward():
            self.grad += out.grad/self.data
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        x_exp, neg_x_exp = math.exp(x), math.exp(-x)
        t = (x_exp - neg_x_exp) / (x_exp + neg_x_exp)
        out = Value(t, (self,))

        def _backward():
            self.grad += (1-t**2)*out._grad
        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        out = Value(max(0.0, x), (self,))

        def _backward():
            self.grad += (x > 0) * out.grad
        out._backward = _backward
        return out

    def leakyrelu(self, alpha: float):
        assert alpha >= 0.0 and alpha <= 1.0, "Probability bounded b/w 0 and 1"
        x = self.data
        out = Value((1 if x > 0 else alpha) * x, (self,))

        def _backward():
            self.grad += (1 if x > 0 else alpha) * out.grad
        out._backward = _backward

    def backward(self):
        topological_list = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in self._prev:
                    build(child)
                topological_list.append(node)
        build(self)

        self.grad = 1.0
        for node in reversed(topological_list):
            node._backward()
