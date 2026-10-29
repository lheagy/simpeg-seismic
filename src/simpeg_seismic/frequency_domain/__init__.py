"""
=========================================================
Frequency Domain (:mod:`simpeg_seismic.frequency_domain`)
=========================================================
.. currentmodule:: simpeg_seismic.frequency_domain

Frequency-domain acoustic (Helmholtz) surveys and simulations, for
full-waveform inversion.

Simulations
===========
.. autosummary::
  :toctree: generated/

  SimulationAcoustic

Fields
======
.. autosummary::
  :toctree: generated/

  FieldsAcoustic

Survey, Sources and Receivers
=============================
.. autosummary::
  :toctree: generated/

  Survey
  sources.PointSource
  receivers.Pressure
"""

from . import receivers, sources
from . import receivers as Rx
from . import sources as Src
from .fields import FieldsAcoustic
from .simulation import SimulationAcoustic
from .survey import Survey

__all__ = [
    "Survey",
    "SimulationAcoustic",
    "FieldsAcoustic",
    "sources",
    "receivers",
    "Src",
    "Rx",
]
