import random
import numpy as np
from backpropagation import Value

class Layer:

    def __init__(self, nin, nout, nonlin=True):
        # weight matrix: shape (nout, nin)
        self.W = [[Value(random.uniform(-1, 1)) for _ in range(nin)] for _ in range(nout)]
        # bias vector: shape (nout,)
        self.b = [Value(0) for _ in range(nout)]
        self.nonlin = nonlin

    def __call__(self, x):
        # Single input: vector of Values
        if isinstance(x[0], (int, float)) or hasattr(x[0], "data"):
            return self._forward_single(x)

        # Batch input: list of samples
        return [self._forward_single(x_i) for x_i in x]

    def _forward_single(self, x):
        out = []
        for w_row, b in zip(self.W, self.b):
            # dot product: w · x + b
            act = sum((wi * xi for wi, xi in zip(w_row, x)), b)
            if self.nonlin:
                act = act.relu()
            out.append(act)
        return out

    def parameters(self):
        # flatten weight matrix + biases
        return [p for row in self.W for p in row] + self.b

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
