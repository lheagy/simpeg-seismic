# simpeg-seismic

Seismic forward simulations that plug into [SimPEG](https://simpeg.xyz).

## Modules

- `simpeg_seismic.straight_ray`: straight-ray traveltime tomography on 2D and
  3D `discretize.TensorMesh` and `discretize.TreeMesh`.

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
