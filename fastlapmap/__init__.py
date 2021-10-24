import sys

from . import ann
from . import spectral

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['ann', 'spectral']})

