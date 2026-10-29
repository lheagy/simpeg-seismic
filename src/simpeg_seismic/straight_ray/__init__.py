"""
Straight-ray traveltime tomography.

Traveltimes are line integrals of slowness along the straight line between
each source and receiver, so the forward problem is linear, ``d = G s``,
where ``G`` holds the length of each ray in each cell. Works on 2D and 3D
``discretize.TensorMesh`` and ``discretize.TreeMesh``.

Classes
-------
Simulation
Survey
PointSource
Traveltime
"""

from . import receivers, sources
from .receivers import Traveltime
from .simulation import Simulation
from .sources import PointSource
from .survey import Survey

__all__ = ["Simulation", "Survey", "PointSource", "Traveltime", "receivers", "sources"]
