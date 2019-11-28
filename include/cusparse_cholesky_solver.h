#ifndef __CUSPARSE_CHOLESKY_SOLVER_H__
#define __CUSPARSE_CHOLESKY_SOLVER_H__

#include <memory>

template <typename T>
class CuSparseCholeskySolver
{
public:

	enum Info
	{
		SUCCESS,
		NUMERICAL_ISSUE
	};

	using Ptr = std::unique_ptr<CuSparseCholeskySolver>;

	static Ptr create(int size = 0);
	virtual void resize(int size) = 0;
	virtual void analyze(int nnz, const int* csrRowPtr, const int* csrColInd) = 0;
	virtual void factorize(const T* A) = 0;
	virtual void solve(const T* b, T* x) = 0;
	virtual void setPermutaion(int size, const int* P) = 0;
	virtual Info info() const = 0;

	virtual ~CuSparseCholeskySolver();
};

#endif // !__CUSPARSE_CHOLESKY_SOLVER_H__
