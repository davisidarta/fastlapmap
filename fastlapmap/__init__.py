import sys

from .ann import NMSlibTransformer as ApproxKNeighbors
from .spectral import LapEigenmap
from .similarities import diffusion_harmonics
from .similarities import fuzzy_simplicial_set_ann as fuzzy_similarity
from .similarities import cknn_graph

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['ApproxKNeighbors', 'LapEigenmap', 'fuzzy_similarity', 'diffusion_harmonics', 'cknn_graph']})

