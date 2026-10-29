from simpeg.survey import BaseSurvey
from simpeg.utils import validate_list_of_types

from .sources import PointSource


class Survey(BaseSurvey):
    """Frequency-domain acoustic survey.

    Sources are grouped by frequency: the simulation factors the system
    matrix once per frequency and solves for all of its sources together.

    Parameters
    ----------
    source_list : list of simpeg_seismic.frequency_domain.sources.PointSource
        The sources, each with its frequency and receivers.
    """

    @property
    def source_list(self):
        """Sources of the survey.

        Returns
        -------
        list of simpeg_seismic.frequency_domain.sources.PointSource
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list", new_list, PointSource, ensure_unique=True
        )
        by_frequency = {}
        for src in self._source_list:
            by_frequency.setdefault(src.frequency, []).append(src)
        self._sources_by_frequency = by_frequency

    @property
    def frequencies(self):
        """Frequencies in the survey, sorted.

        Returns
        -------
        list of float
        """
        return sorted(self._sources_by_frequency)

    @property
    def num_frequencies(self):
        """Number of frequencies in the survey.

        Returns
        -------
        int
        """
        return len(self._sources_by_frequency)

    @property
    def num_sources_by_frequency(self):
        """Number of sources at each frequency.

        Returns
        -------
        dict of {float: int}
        """
        return {f: len(self._sources_by_frequency[f]) for f in self.frequencies}

    def get_sources_by_frequency(self, frequency):
        """Sources at one frequency.

        Parameters
        ----------
        frequency : float

        Returns
        -------
        list of simpeg_seismic.frequency_domain.sources.PointSource
        """
        if frequency not in self._sources_by_frequency:
            raise KeyError(f"No sources at frequency {frequency} Hz.")
        return self._sources_by_frequency[frequency]
