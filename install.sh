#install locally for debugging and such... regular install is just:
# pip install koleksyon 
# pip install 'koleksyon==0.0.7' --force-reinstall
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
python setup.py build_ext --inplace
pip install -e .
