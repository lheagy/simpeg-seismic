# simpeg-seismic

Seismic forward simulations that plug into [SimPEG](https://simpeg.xyz).

## Layout

The package follows the layout of `simpeg.electromagnetics`: the survey,
sources and receivers are shared, and the simulations sit side by side.

```python
import simpeg_seismic as seismic

rx = seismic.receivers.Traveltime(receiver_locations)
src = seismic.sources.PointSource([rx], location=source_location)
survey = seismic.Survey([src])
sim = seismic.SimulationStraightRay(mesh, survey=survey, slownessMap=slowness_map)
```

- `survey`, `sources`, `receivers`: first-arrival traveltime surveys.
- `simulation`:
  - `SimulationStraightRay`: straight-ray traveltimes on 2D and 3D
    `discretize.TensorMesh` and `discretize.TreeMesh`.
- `utils`: ray-cell intersection lengths (`ray_cell_length_matrix`), with an
  optional numba engine.

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
