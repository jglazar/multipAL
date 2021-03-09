# multipAL

multipAL is an active learning package that optimizes material compositions to maximize one or more properties. MultipAL works closely in tandem with the NIST JARVIS DFT database and VASP DFT software. It implements a random forest regressor to make predictions and quantifies the uncertainties using the ForestCI package.

## Installation

First, make sure the dependencies are installed with the correct versions. You can use the `environment.yml` file in the `extras/` directory to recreate a working conda environment. The Unix command is `conda env create -f environment.yml`.

Second, overwrite the `vasp.py` in your environment's `.../lib/python3.8/site-packages/jarvis/tasks/vasp/` directory with the `vasp.py` file provided in the `extras/` directory. This fixes a few issues with the default VASP settings.

You're now ready to use the `multipal` package! Please follow the tutorials in `examples/` to learn more about what the package can do, and how to use it.

This code is made available under the MIT License.
