import math
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # TODO: создать out с правильными _children и _op

        other = Value(other) if isinstance(other, (int, float)) else other
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # TODO
        other = Value(other) if isinstance(other, (int, float)) else other
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "только int/float в показателе"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward
        return out

    def tanh(self):
        # TODO: посчитай t, используй 1 - t**2 в _backward
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out

    def exp(self):
        t = math.exp(self.data)
        out = Value(t, (self,), 'exp')

        def _backward():
            self.grad += out.grad * t

        out._backward = _backward
        return out

    def relu(self):
        if self.data >= 0:
            out = Value(self.data, (self,), 'relu')

            def _backward():
                self.grad += out.grad * 1
        else:
            out = Value(0, (self,), 'relu')

            def _backward():
                self.grad += out.grad * 0

        out._backward = _backward
        return out


    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __truediv__(self, other):
        other = Value(other) if isinstance(other, (int, float)) else other
        return self * (other ** (-1))
    def __radd__(self, other):
        other = Value(other)
        return self + other
    def __rmul__(self, other):
        other = Value(other)
        return self * other
    def __rsub__(self, other):
        other = Value(other)
        return other - self

    def Backward(self):
        topo = []
        visited = set()
        self.grad = 1.0
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                # TODO: куда добавлять v — до рекурсии или после?
                topo.append(v)

        build(self)
        # TODO: база рекурсии + обход
        topo = topo[::-1]

        for el in topo:
            el._backward()

    def zero_grad(self):
        topo = []
        visited = set()
        # TODO: тот же обход, что в backward
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        topo = topo[::-1]
        for v in topo:
            v.grad = 0.0


import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        # TODO: sum(wi*xi) + b, потом tanh
        s = Value(0.0)
        w = self.w
        for i in range(len(w)):
            s += w[i] * x[i]
        s += self.b
        return s.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # TODO
        p = []
        for i in self.neurons:
            p += i.parameters()
        return p

class MLP:
    def __init__(self, nin, nn):
        layers = [Layer(nin, nn[0])]
        for i in range(1, len(nn)):
            layers += [Layer(nn[i-1], nn[i])]
        self.layers = layers

    def __call__(self, x):
        for i in self.layers:
            x = i(x)
        return x

    def parameters(self):
        p = []
        for i in self.layers:
            p += i.parameters()
        return p

import torch

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([-3.0], requires_grad=True)
c = torch.tensor([10.0], requires_grad=True)
d = torch.tensor([-2.0], requires_grad=True)

f = (a * b + c) * d
f.backward()
print(a.grad.item(), b.grad.item(), c.grad.item(), d.grad.item())

# TODO: то же на своём движке, сравнить попарно
A = Value(2.0)
B = Value(-3.0)
C = Value(10.0)
D = Value(-2.0)
f = (A * B + C) * D
f.Backward()
print(A.grad, B.grad, C.grad, D.grad)

# сравнение своего движка с torch на MLP-подобном выражении
import math


x = Value(0.7); w = Value(-1.3); b = Value(0.4)
f = (x * w + b).tanh() * Value(2.0) + x ** 2

def theirs():
    x = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([-1.3], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    out = torch.tanh(x * w + b) * 2.0 + x ** 2
    out.backward()
    return x.grad.item(), w.grad.item(), b.grad.item()

# TODO: прогнать оба, сравнить с допуском 1e-9

f.Backward()

x_grad, w_grad, b_grad = theirs()
dif = x.grad - x_grad, w.grad - w_grad, b.grad - b_grad
print(dif)