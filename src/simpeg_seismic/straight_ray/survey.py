import numpy as np
from simpeg.survey import BaseSurvey


class Survey(BaseSurvey):
    """Straight-ray traveltime survey.

    Parameters
    ----------
    source_list : list of simpeg_seismic.straight_ray.sources.PointSource
        Sources, each with its list of receivers.
    """

    @property
    def ray_endpoints(self):
        """Source and receiver location of every datum.

        Rows are in data order: by source, then receiver, then location.

        Returns
        -------
        source_locations, receiver_locations : (nD, dim) numpy.ndarray
        """
        source_locations, receiver_locations = [], []
        for src in self.source_list:
            for rx in src.receiver_list:
                locations = np.atleast_2d(rx.locations)
                receiver_locations.append(locations)
                source_locations.append(np.tile(src.location, (locations.shape[0], 1)))
        if not receiver_locations:
            return np.empty((0, 0)), np.empty((0, 0))
        return np.vstack(source_locations), np.vstack(receiver_locations)

    def plot(self, ax=None, plot_rays=True):
        """Plot sources, receivers and straight rays.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. For a 3D survey this must be a 3D axes; one is
            created if not given.
        plot_rays : bool, optional
            Whether to draw the straight ray between each source and receiver.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        source_locations, receiver_locations = self.ray_endpoints
        dim = receiver_locations.shape[1]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)

        if plot_rays:
            for a, b in zip(source_locations, receiver_locations):
                ax.plot(*np.c_[a, b], "k-", lw=0.5, alpha=0.5)
        src_xyz = np.vstack([src.location for src in self.source_list])
        ax.plot(*src_xyz.T, "r*", ms=10, ls="", label="sources")
        ax.plot(*receiver_locations.T, "b^", ms=6, ls="", label="receivers")
        return ax
