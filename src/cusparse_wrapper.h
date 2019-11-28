#ifndef __CUSPARSE_WRAPPER_H__
#define __CUSPARSE_WRAPPER_H__

#include <cusparse.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

/* Description: Gather of non-zero elements from dense vector y into
   sparse vector x. */
cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
	int nnz,
	const float *y,
	float *xVal,
	const int *xInd,
	cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
	int nnz,
	const double *y,
	double *xVal,
	const int *xInd,
	cusparseIndexBase_t idxBase);

/*
 * Low level API for GPU Cholesky
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholBufferInfo(
	cusolverSpHandle_t handle,
	int n,
	int nnzA,
	const cusparseMatDescr_t descrA,
	const float *csrValA,
	const int *csrRowPtrA,
	const int *csrColIndA,
	csrcholInfo_t info,
	size_t *internalDataInBytes,
	size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholBufferInfo(
	cusolverSpHandle_t handle,
	int n,
	int nnzA,
	const cusparseMatDescr_t descrA,
	const double *csrValA,
	const int *csrRowPtrA,
	const int *csrColIndA,
	csrcholInfo_t info,
	size_t *internalDataInBytes,
	size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholFactor(
	cusolverSpHandle_t handle,
	int n,
	int nnzA,
	const cusparseMatDescr_t descrA,
	const float *csrValA,
	const int *csrRowPtrA,
	const int *csrColIndA,
	csrcholInfo_t info,
	void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholFactor(
	cusolverSpHandle_t handle,
	int n,
	int nnzA,
	const cusparseMatDescr_t descrA,
	const double *csrValA,
	const int *csrRowPtrA,
	const int *csrColIndA,
	csrcholInfo_t info,
	void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholZeroPivot(
	cusolverSpHandle_t handle,
	csrcholInfo_t info,
	float tol,
	int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholZeroPivot(
	cusolverSpHandle_t handle,
	csrcholInfo_t info,
	double tol,
	int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(
	cusolverSpHandle_t handle,
	int n,
	const float *b,
	float *x,
	csrcholInfo_t info,
	void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(
	cusolverSpHandle_t handle,
	int n,
	const double *b,
	double *x,
	csrcholInfo_t info,
	void *pBuffer);

#endif // !__CUSPARSE_WRAPPER_H__
