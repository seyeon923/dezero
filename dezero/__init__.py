__all__ = ['Variable', 'Function', 'functions']


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]

        while funcs:
            func = funcs.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)

            if x.creator:
                funcs.append(x.creator)


class Function:
    def __call__(self, input: Variable):
        if type(input) is not Variable:
            raise TypeError(f'input type must be {Variable.__name__} type')
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
