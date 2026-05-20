3dgs_io
=======

A Python library for reading and writing 3D Gaussian Splatting data in
`glTF <https://www.khronos.org/gltf/>`_ (``KHR_gaussian_splatting``),
`3D Tiles <https://www.ogc.org/standard/3dtiles/>`_ (OGC),
`SPZ <https://github.com/nianticlabs/spz>`_, and PLY formats.

Features
--------

- **glTF/GLB** save & load with the ``KHR_gaussian_splatting`` extension
- **SPZ compression** via ``KHR_gaussian_splatting_compression_spz_2``
- **3D Tiles** reader/writer with multi-layer support (camera 3DGS / LiDAR 2DGS)
- **LiDAR 2DGS** dedicated I/O for surfel-based Gaussian representations
- **SPZ / PLY** import & export with automatic coordinate system conversion
- **Typed metadata** (``GlbMetadata``) stored in ``asset.extras``

Installation
------------

Requires Python >= 3.10.

.. code-block:: bash

   pip install 3dgs-io

Or with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv add 3dgs-io

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/gltf_io
   api/spz_io
   api/tiles_io
   api/tiles_export
   api/lidar_2dgs
   api/metadata
   api/viewer
