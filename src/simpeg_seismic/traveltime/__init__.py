"""
First-arrival traveltime surveys and simulations.

Survey, sources and receivers
-----------------------------
Survey
sources.PointSource
receivers.Traveltime

Simulations
-----------
SimulationStraightRay
SimulationEikonal
EikonalFields
"""

from . import receivers, simulation, sources
from . import receivers as Rx
from . import sources as Src
from .simulation import EikonalFields, SimulationEikonal, SimulationStraightRay
from .survey import Survey

__all__ = [
    "Survey",
    "SimulationStraightRay",
    "SimulationEikonal",
    "EikonalFields",
    "sources",
    "receivers",
    "Src",
    "Rx",
    "simulation",
]
