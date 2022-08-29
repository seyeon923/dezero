__all__ = ['Variable', 'Function', 'functions', 'using_config',
           'enable_backprob', 'disable_backprob', 'as_variable', 'Config']

__is_simple_core = True

from . import functions  # nopep8

if __is_simple_core:
    from .core_simple import Variable
    from .core_simple import Function
    from .core_simple import Config
    from .core_simple import using_config
    from .core_simple import enable_backprob
    from .core_simple import disable_backprob
    from .core_simple import as_variable
else:
    raise NotImplementedError('core module not implemented!')
