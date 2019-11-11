#include "cusparse_cholesky_solver.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include "device_buffer.h"
#include "cusparse_wrapper.h"

template <typename T>
class SparseMatrixCSR
{

public:

	void init(int m, int n, int nnz)
	{
		m_ = m;
		n_ = n;
		nnz_ = nnz;

		values_.allocate(nnz);
		rowPtr_.allocate(m + 1);
		colInd_.allocate(nnz);
	}

	const T* val() const { return values_.data; }
	T* val() { return values_.data; }

	const int* rowPtr() const { return rowPtr_.data; }
	int* rowPtr() { return rowPtr_.data; }

	const int* colInd() const { return colInd_.data; }
	int* colInd() { return colInd_.data; }

	int rows() const { return m_; }
	int cols() const { return n_; }
	int nnz() const { return nnz_; }

private:

	DeviceBuffer<T> values_;
	DeviceBuffer<int> rowPtr_;
	DeviceBuffer<int> colInd_;
	int m_, n_, nnz_;
};

template <typename T>
class SparseTriangularLinearSolver
{

public:

	void init(cusparseHandle_t handle, const SparseMatrixCSR<T>& A, bool transpose = false)
	{
		handle_ = handle;
		m_ = A.rows();
		nnz_ = A.nnz();

		// create descriptor
		cusparseCreateMatDescr(&desc_);
		cusparseSetMatType(desc_, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(desc_, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(desc_, CUSPARSE_DIAG_TYPE_NON_UNIT);
		cusparseSetMatFillMode(desc_, CUSPARSE_FILL_MODE_LOWER);

		// create info
		cusparseCreateCsrsv2Info(&info_);

		// set operation and policy
		trans_ = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		policy_ = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

		// allocate buffer
		int bufSize;
		cusparseXcsrsv2_bufferSize(handle_, trans_, m_, nnz_, desc_,
			(T*)A.val(), A.rowPtr(), A.colInd(), info_, &bufSize);
		buffer_.allocate(bufSize);
	}

	void analyze(const SparseMatrixCSR<T>& A)
	{
		cusparseXcsrsv2_analysis(handle_, trans_, m_, nnz_, desc_,
			A.val(), A.rowPtr(), A.colInd(), info_, policy_, buffer_.data);
	}

	void solve(const SparseMatrixCSR<T>& A, const T* f, T* x)
	{
		const T alpha = 1;
		cusparseXcsrsv2_solve(handle_, trans_, m_, nnz_, &alpha, desc_,
			A.val(), A.rowPtr(), A.colInd(), info_, f, x, policy_, buffer_.data);
	}

	void destroy()
	{
		cusparseDestroyMatDescr(desc_);
		cusparseDestroyCsrsv2Info(info_);
	}

	~SparseTriangularLinearSolver() { destroy(); }

private:

	cusparseHandle_t handle_;
	cusparseOperation_t trans_;
	cusparseMatDescr_t desc_;
	csrsv2Info_t info_;
	cusparseSolvePolicy_t policy_;
	DeviceBuffer<unsigned char> buffer_;

	int m_, nnz_;
};

template <typename T>
class SparseLDLT
{

public:

	void init(cusparseHandle_t handle, const SparseMatrixCSR<T>& A)
	{
		handle_ = handle;
		m_ = A.rows();
		nnz_ = A.nnz();

		// create descriptor
		cusparseCreateMatDescr(&desc_);
		cusparseSetMatType(desc_, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(desc_, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(desc_, CUSPARSE_DIAG_TYPE_NON_UNIT);

		// create info
		cusparseCreateCsric02Info(&info_);

		// set operation and policy
		policy_ = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

		// allocate buffer
		int bufSize;
		cusparseXcsric02_bufferSize(handle_, m_, nnz_, desc_,
			(T*)A.val(), A.rowPtr(), A.colInd(), info_, &bufSize);
		buffer_.allocate(bufSize);
	}

	void analyze(const SparseMatrixCSR<T>& A)
	{
		cusparseXcsric02_analysis(handle_, m_, nnz_, desc_,
			A.val(), A.rowPtr(), A.colInd(), info_, policy_, buffer_.data);
	}

	void factorize(SparseMatrixCSR<T>& A)
	{
		cusparseXcsric02(handle_, m_, nnz_, desc_,
			A.val(), A.rowPtr(), A.colInd(), info_, policy_, buffer_.data);
	}

	void destroy()
	{
		cusparseDestroyMatDescr(desc_);
		cusparseDestroyCsric02Info(info_);
	}

	~SparseLDLT() { destroy(); }

private:

	cusparseHandle_t handle_;
	cusparseMatDescr_t desc_;
	csric02Info_t info_;
	cusparseSolvePolicy_t policy_;
	DeviceBuffer<unsigned char> buffer_;

	int m_, nnz_;
};

template <typename T>
class SortCOO
{
public:

	void init(cusparseHandle_t handle, int m, int n, int nnz, const int* cooRowInd, const int* cooColInd)
	{
		handle_ = handle;
		m_ = m;
		n_ = n;
		nnz_ = nnz;

		// allocate buffer
		size_t bufSize;
		cusparseXcoosort_bufferSizeExt(handle_, m_, n_, nnz_, cooRowInd, cooColInd, &bufSize);
		buffer_.allocate(bufSize);
		perm_.allocate(nnz);
	}

	void operator()(const T* src, T* dst, int* cooRowInd, int* cooColInd)
	{
		cusparseCreateIdentityPermutation(handle_, nnz_, perm_.data);
		cusparseXcoosortByRow(handle_, m_, n_, nnz_, cooRowInd, cooColInd, perm_.data, buffer_.data);
		cusparseXgthr(handle_, nnz_, src, dst, perm_.data, CUSPARSE_INDEX_BASE_ZERO);
	}

private:

	cusparseHandle_t handle_;
	int m_, n_, nnz_;
	DeviceBuffer<unsigned char> buffer_;
	DeviceBuffer<int> perm_;
};

template <typename T>
class CuSparseCholeskySolverImpl : public CuSparseCholeskySolver<T>
{

public:

	CuSparseCholeskySolverImpl()
	{
	}

	void init(int rows, int cols, int nonzero, const T* A, const int* cooRowInd, const int* cooColInd) override
	{
		// set size
		m = rows;
		n = cols;
		nnz = nonzero;

		// create handle
		cusparseCreate(&handle);

		// allocate memory
		Acsr.init(m, n, nnz);
		d_cooRowInd.allocate(nnz);
		d_f.allocate(n);
		d_x.allocate(n);
		d_z.allocate(n);

		// init solver
		solverL.init(handle, Acsr, false);
		solverLT.init(handle, Acsr, true);
		cholesky.init(handle, Acsr);

		// setup sorting
		d_csrValUS.allocate(nnz);
		d_cooRowIndUS.allocate(nnz);
		d_cooColIndUS.allocate(nnz);

		d_cooRowIndUS.upload(cooRowInd);
		d_cooColIndUS.upload(cooColInd);

		sortCOO.init(handle, m, n, nnz, d_cooRowIndUS.data, d_cooColIndUS.data);

		// analyze
		analyze(A);
	}

	void createCSR(const T* A)
	{
		// create unsorted COO
		d_csrValUS.upload(A);
		d_cooRowIndUS.copyTo(d_cooRowInd);
		d_cooColIndUS.copyTo(Acsr.colInd());

		// sort COO
		sortCOO(d_csrValUS.data, Acsr.val(), d_cooRowInd.data, Acsr.colInd());

		// conver COO to CSR
		cusparseXcoo2csr(handle, d_cooRowInd.data, nnz, m, Acsr.rowPtr(), CUSPARSE_INDEX_BASE_ZERO);
	}

	void analyze(const T* A)
	{
		// copy input data to device memory
		createCSR(A);
		cholesky.analyze(Acsr);
		solverL.analyze(Acsr);
		solverLT.analyze(Acsr);
	}

	void solve(const T* A, const T* f, T* x) override
	{
		createCSR(A);

		d_f.upload(f);

		// M = L * LT
		cholesky.factorize(Acsr);

		// solve L * z = x
		solverL.solve(Acsr, d_f.data, d_z.data);

		// solve LT * y = z
		solverLT.solve(Acsr, d_z.data, d_x.data);

		d_x.download(x);
	}

	void destroy()
	{
		cusparseDestroy(handle);
	}

	~CuSparseCholeskySolverImpl()
	{
		destroy();
	}

private:

	SparseMatrixCSR<T> Acsr;

	DeviceBuffer<int> d_cooRowInd;
	DeviceBuffer<T> d_csrValUS; // unsorted
	DeviceBuffer<int> d_cooRowIndUS; // unsorted
	DeviceBuffer<int> d_cooColIndUS; // unsorted

	DeviceBuffer<T> d_f;
	DeviceBuffer<T> d_x;
	DeviceBuffer<T> d_z;

	cusparseHandle_t handle;
	SparseLDLT<T> cholesky;
	SparseTriangularLinearSolver<T> solverL, solverLT;
	SortCOO<T> sortCOO;

	int m, n, nnz;
};

template<typename T>
typename CuSparseCholeskySolver<T>::Ptr CuSparseCholeskySolver<T>::create()
{
	return std::make_unique<CuSparseCholeskySolverImpl<T>>();
}

template<typename T>
CuSparseCholeskySolver<T>::~CuSparseCholeskySolver()
{
}

template class CuSparseCholeskySolver<double>;
template class CuSparseCholeskySolver<float>;
