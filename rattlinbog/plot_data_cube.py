import argparse
from pathlib import Path

import holoviews as hv
import panel as pn
import param as pm
from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from holoviews.operation.datashader import regrid

from rattlinbog.loaders import load_harmonic_orbits

hv.extension('bokeh')


class Explorer(pm.Parameterized):
    orbit = pm.Integer(0, bounds=(0, 7))

    def view(self, **kwargs):
        return regrid(hv.Image(self.dataset['orbits'].isel(orbit=self.orbit))).opts(cmap='seismic')


def main(file_dataset_root: Path):
    param_file_dataset = gather_files(file_dataset_root, yeoda_naming_convention)
    c1_orbits = load_harmonic_orbits(param_file_dataset, 'SIG0-HPAR-C1')
    explorer = Explorer(dataset=c1_orbits, name="")
    panel = pn.Row(pn.Param(explorer.param, expand_button=False), explorer.view)
    panel.servable()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot parameter dataset")
    parser.add_argument("root", help="Root path of file dataset", type=Path)
    args = parser.parse_args()
    main(args.root)
