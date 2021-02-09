This directory stages the distributions...  don't put stuff in here!

Usage of the toolchain to build and distribute the module is described here:
https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html

We also have some helper scripts at the top level directory that should be run from there.

One important approach for developers is to use the development mode as follows:
$ python setup.py develop
or:
$ pip install -e ./

This way you can easily import the code here into other projects and use the functionality e.g.

