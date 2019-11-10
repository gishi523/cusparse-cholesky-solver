#include "cusparse_wrapper.h"

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
	int *pBufferSizeInBytes)
{
	return
		cusparseScsrsv2_bufferSize(handle,
			transA,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_bufferSize(cusparseHandle_t handle,
	cusparseOperation_t transA,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csrsv2Info_t info,
	int *pBufferSizeInBytes)
{
	return
		cusparseDcsrsv2_bufferSize(handle,
			transA,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			pBufferSizeInBytes);
}

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
	void *pBuffer)
{
	return
		cusparseScsrsv2_analysis(handle,
			transA,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			policy,
			pBuffer);
}

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
	void *pBuffer)
{
	return
		cusparseDcsrsv2_analysis(handle,
			transA,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			policy,
			pBuffer);
}

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
	void *pBuffer)
{
	return
		cusparseScsrsv2_solve(handle,
			transA,
			m,
			nnz,
			alpha,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			f,
			x,
			policy,
			pBuffer);
}

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
	void *pBuffer)
{
	return
		cusparseDcsrsv2_solve(handle,
			transA,
			m,
			nnz,
			alpha,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			f,
			x,
			policy,
			pBuffer);
}

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
	int *pBufferSizeInBytes)
{
	return
		cusparseScsric02_bufferSize(handle,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_bufferSize(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	int *pBufferSizeInBytes)
{
	return
		cusparseDcsric02_bufferSize(handle,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_analysis(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	const float *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer)
{
	return
		cusparseScsric02_analysis(handle,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			policy,
			pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_analysis(cusparseHandle_t handle,
	int m,
	int nnz,
	const cusparseMatDescr_t descrA,
	const double *csrSortedValA,
	const int *csrSortedRowPtrA,
	const int *csrSortedColIndA,
	csric02Info_t info,
	cusparseSolvePolicy_t policy,
	void *pBuffer)
{
	return
		cusparseDcsric02_analysis(handle,
			m,
			nnz,
			descrA,
			csrSortedValA,
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			policy,
			pBuffer);
}

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
	void *pBuffer)
{
	return
		cusparseScsric02(handle,
			m,
			nnz,
			descrA,
			csrSortedValA_valM,
			/* matrix A values are updated inplace
			   to be the preconditioner M values */
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			policy,
			pBuffer);
}

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
	void *pBuffer)
{
	return
		cusparseDcsric02(handle,
			m,
			nnz,
			descrA,
			csrSortedValA_valM,
			/* matrix A values are updated inplace
			   to be the preconditioner M values */
			csrSortedRowPtrA,
			csrSortedColIndA,
			info,
			policy,
			pBuffer);
}

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
