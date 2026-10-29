"""
======================================
SimPEG Seismic (:mod:`simpeg_seismic`)
======================================
.. currentmodule:: simpeg_seismic

Seismic simulations for SimPEG, usually imported as
``import simpeg_seismic as seismic``. Like ``simpeg.electromagnetics``, each
kind of data has its own subpackage with its survey, sources, receivers and
simulations.

.. autosummary::
  :toctree: generated/

  traveltime
  frequency_domain
  acoustic
  utils
"""

from . import acoustic, frequency_domain, traveltime, utils

__version__ = "0.0.1.dev0"

__all__ = ["traveltime", "frequency_domain", "acoustic", "utils"]
