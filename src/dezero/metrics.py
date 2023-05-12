__all__ = ['accuracy']

from . import __is_simple_core

if __is_simple_core:
    from .core_simple import Variable, as_variable
else:
    from .core import Variable, as_variable


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=-1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return as_variable(acc)
