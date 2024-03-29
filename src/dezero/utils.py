import tempfile
import os
import traceback
import sys
import subprocess
import urllib.request

from string import Template
from io import StringIO

import numpy as np

from . import Variable, Function

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
        id=id(f), label=f.name))

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


def plot_dot_graph(output: Variable, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose=verbose)

    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as f:
        f.write(dot_graph)

    try:
        ext = os.path.splitext(to_file)[1][1:]
        subprocess.run(['dot', f.name, '-T', ext, '-o', to_file], shell=True,
                       stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as ex:
        stderr = ex.stderr

        if isinstance(stderr, bytes):
            stderr = ex.stderr.decode('utf-8')

        print(stderr, file=sys.stderr)
        print(
            f'Failed to run command "{ex.cmd}"(return code = {ex.returncode})', file=sys.stderr)
    except Exception:
        traceback.print_exc()
    finally:
        os.unlink(f.name)


def sum_to(x: np.ndarray, shape) -> np.ndarray:
    """Sum elements along axes to output an array of a given shape.
    Args:
        `x` (ndarray): Input array.
        `shape`: desired shape of an output array
    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


# =============================================================================
# download function
# Copied from https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/dezero/utils.py
# =============================================================================
def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.
    The file at the `url` is downloaded to the `~/.dezero`.
    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.
    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


if __name__ == '__main__':
    def goldstein(x, y):
        return (1 + (x + y + 1) ** 2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
            (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))

    x = Variable(1.0, name='x')
    y = Variable(1.0, name='y')
    z = goldstein(x, y)
    z.name = 'z'
    z.backward()

    print(get_dot_graph(z))
    plot_dot_graph(z, verbose=False, to_file='goldstein.png')
