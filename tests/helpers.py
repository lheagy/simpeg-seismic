"""Helpers shared by the tests (on the import path via ``pythonpath`` in pyproject)."""

import numpy as np
from discretize.tests import check_derivative

import simpeg_seismic as seismic


def make_survey(source_locations, receiver_locations):
    """Survey where every source records at every receiver location."""
    receivers = [seismic.receivers.Traveltime(np.atleast_2d(receiver_locations))]
    return seismic.Survey(
        [
            seismic.sources.PointSource(receiver_list=receivers, location=loc)
            for loc in source_locations
        ]
    )


def assert_derivative(sim, model, dx):
    """``Jvec`` passes SimPEG's second-order derivative check along ``dx``."""
    assert check_derivative(
        lambda x: (sim.dpred(x), lambda v: sim.Jvec(x, v)),
        model,
        dx=dx,
        num=4,
        plotIt=False,
        random_seed=0,
    )


def assert_adjoint(sim, model, rng, f=None, rtol=1e-10):
    """``Jtvec`` is the transpose of ``Jvec``: w . J v == v . J^T w."""
    v = rng.standard_normal(model.size)
    w = rng.standard_normal(sim.survey.nD)
    np.testing.assert_allclose(
        w @ sim.Jvec(model, v, f=f), v @ sim.Jtvec(model, w, f=f), rtol=rtol
    )
