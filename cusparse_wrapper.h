#ifndef __CUSPARSE_GENERIC_H__
#define __CUSPARSE_GENERIC_H__

#include <cusparse.h>

/* Description: Solution of triangular linear system op(A) * x = alpha * f,
   where A is a sparse matrix in CSR storage format, rhs f and solution y
   are dense vectors. This routine implements algorithm 1 for this problem.
   Also, it provides a utility function to query size of buffer used. */
cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_bufferSize(cusparseHandle_t handle,
	cusparseOperation_t transA,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	float *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csrsv2Info_t info,
	int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_bufferSize(cusparseHandle_t handle,
	cusparseOperation_t transA,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csrsv2Info_t info,
	int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_analysis(cusparseHandle_t handle,
	cusparseOperation_t transA,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	const float *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csrsv2Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_analysis(cusparseHandle_t handle,
	cusparseOperation_t transA,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	const double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csrsv2Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_solve(cusparseHandle_t handle,
	cusparseOperation_t transA,
	int m,
	int nnz,
	const float *alpha,
	const cusparseMatDescr_t descrA,
	const float *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csrsv2Info_t info,
	const float *f,
	float *x,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_solve(cusparseHandle_t handle,
	cusparseOperation_t transA,
	int m,
	int nnz,
	const double *alpha,
	const cusparseMatDescr_t descrA,
	const double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csrsv2Info_t info,
	const double *f,
	double *x,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   of the matrix A stored in CSR format based on the information in the opaque
   structure info that was obtained from the analysis phase (csrsv2_analysis).
   This routine implements algorithm 2 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseXcsric02_bufferSize(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	float *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_bufferSize(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_analysis(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	const float *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_analysis(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	const double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcsric02(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	float *csrSortedValA_valM,
	/* matrix A values are updated inplace
	   to be the preconditioner M values */
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcsric02(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	double *csrSortedValA_valM,
	/* matrix A values are updated inplace
	   to be the preconditioner M values */
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer);

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

#endif // !__CUSPARSE_GENERIC_H__
