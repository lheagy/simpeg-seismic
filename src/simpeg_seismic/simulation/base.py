from discretize import TensorMesh, TreeMesh
from simpeg import props
from simpeg.simulation import BaseSimulation
from simpeg.utils import validate_type

from ..survey import Survey


class BaseTraveltimeSimulation(BaseSimulation):
    """Base class for first-arrival traveltime simulations.

    Holds what every traveltime simulation shares: the slowness model, the
    mesh and the survey. Subclasses declare the mesh types they support in
    ``_mesh_types`` and clear their cached quantities in ``_reset``.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        A 2D or 3D mesh of one of the types in ``_mesh_types``.
    survey : simpeg_seismic.Survey
        The survey.
    slowness : numpy.ndarray, optional
        Slowness (s/m).
    slownessMap : simpeg.maps.IdentityMap, optional
        Mapping from the model to slowness. Use a
        :class:`simpeg.maps.ReciprocalMap` to invert for velocity.
    """

    slowness, slownessMap, slownessDeriv = props.Invertible("Slowness (s/m)")

    _mesh_types = (TensorMesh, TreeMesh)

    def __init__(
        self,
        mesh,
        survey=None,
        slowness=None,
        slownessMap=None,
        **kwargs,
    ):
        self.mesh = mesh
        super().__init__(survey=survey, **kwargs)
        self.slowness = slowness
        self.slownessMap = slownessMap

    @property
    def mesh(self):
        """Mesh the slowness is defined on.

        Returns
        -------
        discretize.base.BaseMesh
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        value = validate_type("mesh", value, self._mesh_types, cast=False)
        if value.dim not in (2, 3):
            raise ValueError(
                f"{type(self).__name__} mesh must be 2D or 3D, "
                f"received a {value.dim}D mesh."
            )
        self._mesh = value
        self._reset()

    @BaseSimulation.survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        BaseSimulation.survey.fset(self, value)
        self._reset()

    def _reset(self):
        """Clear quantities that depend on the mesh or survey."""

    def dpred(self, m=None, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(self.model if m is None else m)
        return f
