import random 
from engine import Value
from engine import draw_dot

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(0,nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum( (wi*xi for wi,xi in zip(self.w, x)), self.b)

        out = act.tanh()

        return out
        
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons ]
        return out[0] if len(out) == 1 else out

        
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        

    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



n = MLP(3, [3,4,4,1])


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]  # desired targets



for k in range(20):

    for p in n.parameters():
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # backward pass
        #the parameter's grad is accumulate so we need to zero the grad
        n.zero_grad()
        loss.backward()

        # update
        p.data += 0.05 * p.grad * (-1)
    print(k, loss)
print(ypred)


