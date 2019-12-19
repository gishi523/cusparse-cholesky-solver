# cusparse-cholesky-solver
A sample code for sparse cholesky solver with cuSPARSE and cuSOLVER library

## Description
- A sample code for sparse cholesky solver with cuSPARSE and cuSOLVER library
- It solves sparse linear system with positive definite matrix using cholesky decomposition


## References
- [The API reference guide for cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)
- [The API reference guide for cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html)

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

### simple_cholesky sample

```
./samples/simple_cholesky/sample_simple_cholesky
```

### large_cholesky sample

```
./samples/large_cholesky/sample_large_cholesky matrix-market-file
```

- For example, download the [ND/ND6k](https://sparse.tamu.edu/ND/nd6k) matrix from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu)
- Unpack the nd6k.mtx and pass to the program


```
./samples/large_cholesky/sample_large_cholesky nd6k.mtx
```
