"""
=============================================
Traveltime (:mod:`simpeg_seismic.traveltime`)
=============================================
.. currentmodule:: simpeg_seismic.traveltime

First-arrival traveltime surveys and simulations.

Simulations
===========
.. autosummary::
  :toctree: generated/

  SimulationStraightRay
  SimulationEikonal

Fields
======
.. autosummary::
  :toctree: generated/

  EikonalFields

Survey, Sources and Receivers
=============================
.. autosummary::
  :toctree: generated/

  Survey
  sources.PointSource
  receivers.Traveltime

Base Classes
============
.. autosummary::
  :toctree: generated/

  simulation.BaseTraveltimeSimulation
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
