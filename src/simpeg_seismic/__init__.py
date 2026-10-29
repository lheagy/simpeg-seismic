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
SimulationEikonal
"""

from . import receivers, simulation, sources, utils
from . import receivers as Rx
from . import sources as Src
from .simulation import SimulationEikonal, SimulationStraightRay
from .survey import Survey

__version__ = "0.0.1.dev0"

__all__ = [
    "Survey",
    "SimulationStraightRay",
    "SimulationEikonal",
    "sources",
    "receivers",
    "Src",
    "Rx",
    "simulation",
    "utils",
]
