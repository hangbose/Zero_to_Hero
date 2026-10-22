import math
class Value:
    def __init__(self, data, _children=(), _op='',label=''):
        self.data = data
        self.grad = 0
        self.label = label 
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other  if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self, other),'+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
        return self + (-other)
    def __rmul__(self, other):
        return self * other 
    def __mul__(self, other):
        other = other  if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**(-1))

    def __rtruediv__(self, other):  #other / self
        return  other * (self ** (-1))

    def __radd__(self, other):
        return self + other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int or float now"
        out = Value(math.pow(self.data, other), (self, ), f'**{other}' )
        def _backward():
            self.grad += other *( self.data ** (other - 1) )* out.grad

        out._backward = _backward
        return out 


    def relu(self):
        x = self.data
        out = Value(0 if x < 0 else x, (self, ), 'relu')
        def _backward():
            self.grad += (x > 0) * out.grad

        out._backward = _backward
        return out


    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1) 
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t*t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += math.exp(x) * out.grad

        out._backward = _backward
        return out

    def __repr__(self):
        return f"{self.label}  Value(data={self.data})"
    
    def backward(self):

        topo = []
        visited = set()
        def bulid_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    bulid_topo(child)

                topo.append(v)

        bulid_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()

        

from graphviz import Digraph


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges


def draw_dot(root):
    dot = Digraph(
        format='svg',
        graph_attr={'rankdir': 'LR'}
    )

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        # for any value in the graph,
        # create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }"
                  % (n.label, n.data, n.grad),
            shape='record'
        )

        if n._op:
            # if this value is a result of some operation,
            # create an op node for it
            dot.node(
                name=uid + n._op,
                label=n._op
            )

            # and connect this node to it
            dot.edge(
                uid + n._op,
                uid
            )

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(
            str(id(n1)),
            str(id(n2)) + n2._op
        )

    return dot

def main():
    x1 = Value(2, label='x1')
    x2 = Value(0, label='x2')


    w1 = Value(-3, label='w1')
    w2 = Value(1, label='w2')

    b = Value(6.8813735870195432, label='b')

    x1w1 = x1 * w1; x1w1.label = "x1*w1"
    x2w2 = x2 * w2; x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1+x2w2'

    n = x1w1x2w2 +  b; n.label= 'n'
    #-------------------------
    e = (2*n).exp()
    o =(e -1) / (e + 1)

    #------------
    o.label = 'o'

    o.backward()
    dot = draw_dot(o)

    path = dot.render("micrograd_graph", view=False)
    print(path)

if __name__ == "__main__":
    main()