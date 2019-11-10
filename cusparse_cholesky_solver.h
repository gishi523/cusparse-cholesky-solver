#ifndef __CUSPARSE_CHOLESKY_SOLVER_H__
#define __CUSPARSE_CHOLESKY_SOLVER_H__

#include <memory>

template <typename T>
class CuSparseCholeskySolver
{
public:

	using Ptr = std::unique_ptr<CuSparseCholeskySolver>;

	static Ptr create();
	virtual void init(int rows, int cols, int nnz, const T* A, const int* cooRowInd, const int* cooColInd) = 0;
	virtual void solve(const T* A, const T* f, T* x) = 0;

	virtual ~CuSparseCholeskySolver();
};

#endif // !__CUSPARSE_CHOLESKY_SOLVER_H__
