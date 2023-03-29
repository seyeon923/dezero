__all__ = ['Variable', 'Function', 'functions', 'using_config',
           'enable_backprob', 'disable_backprob', 'as_variable', 'Config']

__is_simple_core = False

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
    from .core import Variable
    from .core import Function
    from .core import Config
    from .core import using_config
    from .core import enable_backprob
    from .core import disable_backprob
    from .core import as_variable
