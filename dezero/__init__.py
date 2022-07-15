__all__ = ['Variable', 'Function', 'functions']


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input: Variable):
        if type(input) is not Variable:
            raise TypeError(f'input type must be {Variable.__name__} type')
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()
