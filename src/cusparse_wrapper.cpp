#include "cusparse_wrapper.h"

/* Description: Gather of non-zero elements from dense vector y into
   sparse vector x. */
cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
	int nnz,
	const float *y,
	float *xVal,
	const int *xInd,
	cusparseIndexBase_t idxBase)
{
	return
		cusparseSgthr(handle,
			nnz,
			y,
			xVal,
			xInd,
			idxBase);
}

/*
 * Low level API for GPU Cholesky
 *
 */
cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
	int nnz,
	const double *y,
	double *xVal,
	const int *xInd,
	cusparseIndexBase_t idxBase)
{
	return
		cusparseDgthr(handle,
			nnz,
			y,
			xVal,
			xInd,
			idxBase);
}

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
	size_t *workspaceInBytes)
{
	return
		cusolverSpScsrcholBufferInfo(
			handle,
			n,
			nnzA,
			descrA,
			csrValA,
			csrRowPtrA,
			csrColIndA,
			info,
			internalDataInBytes,
			workspaceInBytes);
}

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
	size_t *workspaceInBytes)
{
	return
		cusolverSpDcsrcholBufferInfo(
			handle,
			n,
			nnzA,
			descrA,
			csrValA,
			csrRowPtrA,
			csrColIndA,
			info,
			internalDataInBytes,
			workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholFactor(
	cusolverSpHandle_t handle,
	int n,
	int nnzA,
	const cusparseMatDescr_t descrA,
	const float *csrValA,
	const int *csrRowPtrA,
	const int *csrColIndA,
	csrcholInfo_t info,
	void *pBuffer)
{
	return
		cusolverSpScsrcholFactor(handle,
			n,
			nnzA,
			descrA,
			csrValA,
			csrRowPtrA,
			csrColIndA,
			info,
			pBuffer);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholFactor(
	cusolverSpHandle_t handle,
	int n,
	int nnzA,
	const cusparseMatDescr_t descrA,
	const double *csrValA,
	const int *csrRowPtrA,
	const int *csrColIndA,
	csrcholInfo_t info,
	void *pBuffer)
{
	return
		cusolverSpDcsrcholFactor(handle,
			n,
			nnzA,
			descrA,
			csrValA,
			csrRowPtrA,
			csrColIndA,
			info,
			pBuffer);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholZeroPivot(
	cusolverSpHandle_t handle,
	csrcholInfo_t info,
	float tol,
	int *position)
{
	return
		cusolverSpScsrcholZeroPivot(
			handle,
			info,
			tol,
			position);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholZeroPivot(
	cusolverSpHandle_t handle,
	csrcholInfo_t info,
	double tol,
	int *position)
{
	return
		cusolverSpDcsrcholZeroPivot(
			handle,
			info,
			tol,
			position);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(
	cusolverSpHandle_t handle,
	int n,
	const float *b,
	float *x,
	csrcholInfo_t info,
	void *pBuffer)
{
	return
		cusolverSpScsrcholSolve(
			handle,
			n,
			b,
			x,
			info,
			pBuffer);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(
	cusolverSpHandle_t handle,
	int n,
	const double *b,
	double *x,
	csrcholInfo_t info,
	void *pBuffer)
{
	return
		cusolverSpDcsrcholSolve(
			handle,
			n,
			b,
			x,
			info,
			pBuffer);
}
