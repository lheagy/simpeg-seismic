from simpeg.fields import Fields


class FieldsAcoustic(Fields):
    """Fields of :class:`simpeg_seismic.frequency_domain.SimulationAcoustic`.

    Stores the complex pressure ``pSolution`` on the nodes, one column per
    source.

    .. code-block:: python

        f = simulation.fields(m)
        p = f[:, "p"]                         # (n_nodes, n_sources), complex
        p_first = f[survey.source_list[0], "p"]
    """

    knownFields = {"pSolution": "N"}
    aliasFields = {"p": ["pSolution", "N", "_p"]}
    dtype = complex

    def _p(self, p_solution, source_list):
        return p_solution
