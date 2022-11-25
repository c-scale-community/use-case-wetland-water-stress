from pathlib import Path

import re
from setuptools import setup, find_packages

HERE = Path(__file__).parent
_version_file_contents = (HERE / "rattlinbog" / "version.py").read_text()
matched = re.search(r'"(.*)"', _version_file_contents)
VERSION = matched.group(1) if matched is not None else "UNKNOWN VERSION"

setup(
    name='rattlinbog',
    long_description_content_type='text/markdown',
    packages=find_packages(include=['rattlinbog', 'rattlinbog.*']),
    install_requires=["xarray", "rioxarray", "rasterio", "eotransform", "eotransform-xarray", "eotransform-pandas",
                      "geopandas", "jupyter", "numba", "numpy", "matplotlib", "dask", "distributed", "geopathfinder",
                      "equi7grid", "affine", "zarr"],
    extras_require={"test": ['pytest', 'pytest-cov'], "views": ['holoviews', 'datashader', 'panel', 'param']},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6'
)
