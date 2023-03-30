from . import Layer
from . import utils, functions, layers


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_outptu_sizes, activation=functions.sigmoid):
        super().__init__()
        self.__activations = activation
        self.__layers = []

        for i, out_size in enumerate(fc_outptu_sizes):
            l = layers.Linear(out_size)
            setattr(self, f'l{i}', l)
            self.__layers.append(l)

    def forward(self, x):
        for l in self.__layers[:-1]:
            x = self.__activations(l(x))
        return self.__layers[-1](x)
