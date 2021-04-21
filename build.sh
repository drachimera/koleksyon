#remove the old build
rm -rf ./dist/*
#build the tar in the dist folder
python setup.py sdist
python setup.py build_ext -i
#build the wheels - again will be installed to the dist folder
python setup.py bdist_wheel
