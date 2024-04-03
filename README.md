# cupy-nanobind-example

A minimal demo of a Python project that uses scikit-build-core to compile a CUDA CPython extension that accepts a cupy array via nanobind's DLPack support. This is possibly the most technologies I've strung together in a single sentence.

## Build and install (basic)
```console
$ pip install .
```

## Build and install (editable)
```console
$ pip install scikit-build-core nanobind ninja
$ pip install -e . --no-build-isolation
```

## Test
```
$ python test.py
```
