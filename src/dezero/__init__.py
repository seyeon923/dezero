__all__ = ['Variable', 'Function', 'Parameter', 'Layer', 'Model', 'Optimizer',
           'using_config', 'enable_backprob', 'disable_backprob', 'as_variable', 'Config',
           'functions', 'layers', 'models', 'optimizers']

__is_simple_core = False

if __is_simple_core:
    from .core_simple import Variable
    from .core_simple import Function
    from .core_simple import Config
    from .core_simple import using_config
    from .core_simple import enable_backprob
    from .core_simple import disable_backprob
    from .core_simple import as_variable
else:
    from .core import Variable
    from .core import Function
    from . import functions
    from .core import Config
    from .core import using_config
    from .core import enable_backprob
    from .core import disable_backprob
    from .core import as_variable
    from .core import Parameter
    from .layers import Layer
    from . import layers
    from .models import Model
    from . import models
    from .optimizers import Optimizer
    from . import optimizers

    from .core import _setup_variable
    _setup_variable()
