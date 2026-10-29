# simpeg-seismic

Seismic forward simulations that plug into [SimPEG](https://simpeg.xyz).

## Modules

- `simpeg_seismic.straight_ray`: straight-ray traveltime tomography on 2D and
  3D `discretize.TensorMesh` and `discretize.TreeMesh`.

## Install for development

```bash
pip install -e ".[test]"
pytest
```
