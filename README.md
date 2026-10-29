# simpeg-seismic

Seismic forward simulations that plug into [SimPEG](https://simpeg.xyz).

## Layout

The package follows the layout of `simpeg.electromagnetics`: each kind of
data has its own subpackage with its survey, sources, receivers and
simulations, and shared code lives in `utils`.

```python
from simpeg_seismic import traveltime

rx = traveltime.receivers.Traveltime(receiver_locations)
src = traveltime.sources.PointSource([rx], location=source_location)
survey = traveltime.Survey([src])
sim = traveltime.SimulationEikonal(mesh, survey=survey, slownessMap=slowness_map)
```

- `traveltime`: first-arrival traveltimes.
  - `Survey`, `sources.PointSource`, `receivers.Traveltime`.
  - `SimulationStraightRay`: straight-ray traveltimes on 2D and 3D
    `discretize.TensorMesh` and `discretize.TreeMesh`.
  - `SimulationEikonal`: first-arrival traveltimes from the factored eikonal
    equation (first-order Godunov upwind scheme, fast sweeping; exact in a
    homogeneous medium), with exact discrete adjoint-state sensitivities, on
    2D and 3D `discretize.TensorMesh`. Topography through
    `simpeg.maps.InjectActiveCells`. Requires numba.
- `utils`: ray-cell intersection lengths (`ray_cell_length_matrix`, with an
  optional numba engine), the eikonal kernels and mesh helpers.

Planned: `frequency_domain` (Helmholtz, FWI) and `time_domain` (acoustic wave
equation), sharing acoustic operators.

## Examples

The [notebooks](notebooks) directory has forward simulation and inversion
examples:

- `straight_ray_2d_crosswell.ipynb`: 2D crosswell tomography on a
  `TensorMesh`, inverting for slowness.
- `straight_ray_3d_crosshole.ipynb`: 3D tomography between four boreholes and
  a surface array on a `TreeMesh`, inverting for velocity.

## Install for development

```bash
pip install -e ".[test]"
pytest
```

Formatting (black) and linting (flake8) use the same settings as SimPEG and
run as a pre-commit hook:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
