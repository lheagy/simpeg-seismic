from simpeg.survey import BaseRx


class Traveltime(BaseRx):
    """First-arrival traveltime receivers.

    Parameters
    ----------
    locations : (n_loc, dim) numpy.ndarray
        Receiver locations. Each location gives one datum per source.
    """
