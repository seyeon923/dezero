from . import utils
from .layers import Layer


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        utils.plot_dot_graph(y, verbose=True, to_file=to_file)
