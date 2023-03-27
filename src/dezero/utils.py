from . import Variable, Function

import numpy as np
from string import Template
from io import StringIO

__DOT_VAR_TEMPLATE = Template(
    '${id} [label="${label}", color=orange, style=filled]\n')
__DOT_FUNC_TEMPLATE = Template(
    '${id} [label="${label}", color=lightblue, style=filled, shape=box]\n')
__DOT_EDGE_TEMPLATE = Template('${e1} -> ${e2}\n')


def _dot_var(v: Variable, verbose=False):
    label = '' if v.name is None else v.name

    if verbose and v.data is not None:
        label += f'{": " if v.name is not None else ""}{v.shape} {v.dtype}'

    return __DOT_VAR_TEMPLATE.substitute(id=id(v), label=label)


def _dot_func(f: Function):
    sio = StringIO()

    sio.write(__DOT_FUNC_TEMPLATE.substitute(
        id=id(f), label=f.__class__.__name__))

    for x in f.inputs:
        sio.write(__DOT_EDGE_TEMPLATE.substitute(e1=id(x), e2=id(f)))
    for y in f.outputs:
        sio.write(__DOT_EDGE_TEMPLATE.substitute(e1=id(f), e2=id(y())))

    return sio.getvalue()


def get_dot_graph(output: Variable, verbose=True):
    sio = StringIO()

    funcs: list[Function] = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    sio.write(_dot_var(output, verbose=verbose))
    if output.creator is not None:
        add_func(output.creator)

    while funcs:
        func = funcs.pop()
        sio.write(_dot_func(func))

        for input in func.inputs:
            sio.write(_dot_var(input, verbose=verbose))

            if input.creator is not None:
                add_func(input.creator)

    return f'digraph g {{\n{sio.getvalue()}}}'


if __name__ == '__main__':
    x0 = Variable(1.0, name='x0')
    x1 = Variable(1.0, name='x1')

    y = x0 + x1
    y.name = 'y'

    print(get_dot_graph(y))
