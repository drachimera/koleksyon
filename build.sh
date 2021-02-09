#build the tar in the dist folder
python setup.py sdist
#build the wheels - again will be installed to the dist folder
python setup.py bdist_wheel
