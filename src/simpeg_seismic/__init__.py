"""
Seismic forward simulations for SimPEG.

Usually imported as ``import simpeg_seismic as seismic``. Like
``simpeg.electromagnetics``, each kind of data has its own subpackage with
its survey, sources, receivers and simulations.

Subpackages
-----------
traveltime
    First-arrival traveltimes: straight-ray and eikonal simulations.
utils
    Ray geometry, eikonal kernels and mesh helpers.
"""

from . import traveltime, utils

__version__ = "0.0.1.dev0"

__all__ = ["traveltime", "utils"]
