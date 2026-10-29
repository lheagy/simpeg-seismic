from simpeg.survey import BaseSrc


class PointSource(BaseSrc):
    """Seismic point source.

    Parameters
    ----------
    receiver_list : list of simpeg_seismic.receivers.Traveltime
        Receivers that record this source.
    location : (dim,) numpy.ndarray
        Location of the source.
    """

    def __init__(self, receiver_list=None, location=None):
        if location is None:
            raise TypeError("PointSource requires a location.")
        super().__init__(receiver_list=receiver_list, location=location)
