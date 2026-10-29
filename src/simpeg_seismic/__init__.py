"""
Seismic forward simulations for SimPEG.

Usually imported as ``import simpeg_seismic as seismic``.

Survey
------
Survey
sources.PointSource
receivers.Traveltime

Simulations
-----------
SimulationStraightRay
"""

from . import receivers, simulation, sources, utils
from . import receivers as Rx
from . import sources as Src
from .simulation import SimulationStraightRay
from .survey import Survey

__version__ = "0.0.1.dev0"

__all__ = [
    "Survey",
    "SimulationStraightRay",
    "sources",
    "receivers",
    "Src",
    "Rx",
    "simulation",
    "utils",
]
