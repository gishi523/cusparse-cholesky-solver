#include <iostream>
#include <fstream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "cusparse_cholesky_solver.h"

using Scalar = double;
using SparseMatrix = Eigen::SparseMatrix<Scalar>;
using Triplet = Eigen::Triplet<Scalar>;
using Vector = Eigen::Matrix<Scalar, -1, 1>;

std::vector<Triplet> getnerateTriplets(int m, int n, Scalar diagVal = 4, Scalar nonDiagVal = -1)
{
	std::vector<Triplet> triplets;
	for (int i = 0; i < m; i++)
	{
		triplets.push_back(Triplet(i, i, diagVal));

		if (i > 0)
			triplets.push_back(Triplet(i, i - 1, nonDiagVal));

		if (i < n - 1)
			triplets.push_back(Triplet(i, i + 1, nonDiagVal));
	}
	return triplets;
}

void convertToCOO(const std::vector<Triplet>& triplets, std::vector<Scalar>& cooVal, std::vector<int>& cooRowInd, std::vector<int>& cooColInd)
{
	const int nnz = static_cast<int>(triplets.size());
	cooVal.resize(nnz);
	cooRowInd.resize(nnz);
	cooColInd.resize(nnz);

	for (int i = 0; i < nnz; i++)
	{
		const auto& t = triplets[i];
		const Scalar val = t.value();
		const int r = t.row();
		const int c = t.col();
		cooVal[i] = val;
		cooRowInd[i] = r;
		cooColInd[i] = c;
	}
}

int main()
{
	constexpr int M = 64;
	constexpr int N = M;
	const auto triplets = getnerateTriplets(M, N, 4, -1);
	const int nnz = static_cast<int>(triplets.size());

	// generate symmetric sparse matrix
	SparseMatrix A;
	A.resize(M, N);
	A.setFromTriplets(std::begin(triplets), std::end(triplets));

	// generate x
	const Vector x = Vector::Random(N);

	// generate b
	const Vector b = A * x;

	// estimate x with Eigen
	Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper> cholesky;
	cholesky.analyzePattern(A);
	cholesky.factorize(A);
	Vector xhatCPU = cholesky.solve(b);

	// estimate x with cuSparse
	std::vector<Scalar> cooVal;
	std::vector<int> cooRowInd, cooColInd;
	convertToCOO(triplets, cooVal, cooRowInd, cooColInd);

	auto solver = CuSparseCholeskySolver<Scalar>::create();
	solver->init(M, N, nnz, cooVal.data(), cooRowInd.data(), cooColInd.data());

	Vector xhatGPU(N);
	solver->solve(cooVal.data(), b.data(), xhatGPU.data());

	// put errros
	std::cout << "error CPU: " << (x - xhatCPU).squaredNorm() << std::endl;
	std::cout << "error GPU: " << (x - xhatGPU).squaredNorm() << std::endl;

	return 0;
}
