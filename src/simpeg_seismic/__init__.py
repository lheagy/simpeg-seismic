"""
Seismic forward simulations for SimPEG.

Usually imported as ``import simpeg_seismic as seismic``. Like
``simpeg.electromagnetics``, each kind of data has its own subpackage with
its survey, sources, receivers and simulations.

Subpackages
-----------
traveltime
    First-arrival traveltimes: straight-ray and eikonal simulations.
frequency_domain
    Frequency-domain acoustic (Helmholtz) simulation, for FWI.
acoustic
    Discrete acoustic operators and PML, shared by the wave-equation
    simulations.
utils
    Ray geometry, eikonal kernels, mesh helpers and source wavelets.
"""

from . import acoustic, frequency_domain, traveltime, utils

__version__ = "0.0.1.dev0"

__all__ = ["traveltime", "frequency_domain", "acoustic", "utils"]
