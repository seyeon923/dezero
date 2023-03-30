__all__ = ['dz', 'functions', 'layers']
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import dezero as dz  # nopep8
import dezero.layers as layers  # nopep8
import dezero.functions as functions  # nopep8
