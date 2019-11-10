# cusparse-cholesky-solver
A sample code for sparse cholesky solver with cuSPARSE library

## Description
- A sample code for sparse cholesky solver with cuSPARSE library
- It solves sparse linear system with positive definite matrix using cholesky decomposition


## References
- [The API reference guide for cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)

## Requirement
- CUDA
- Eigen

## How to build
```
$ git clone https://github.com/gishi523/cusparse-cholesky-solver.git
$ cd cusparse-cholesky-solver
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to run
```
./cusparse_cholesky_solver
```
