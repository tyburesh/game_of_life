Conway's Game of Life implemented using Python. I have used a toroidal geometry to simulate an infinite plane.
Two versions are implemented here: a CPU version and a GPU verion using PyCUDA.
The GPU version is designed to be run on Mesabi, a supercomputer belonging to the Minnesota Supercomputer Institute.

Installation of PyCUDA (can be on any Mesabi node):
> module load python2
> conda create -n pycuda3 python=3 numpy
> source activate pycuda3
> module load cuda
> pip install pycuda

Usage (must be on a k40 node):
> module load python2
> source activate pycuda3
> module load cuda